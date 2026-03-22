"""Training loop for score-based diffusion with optional spectral noise shaping."""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from graph_fans.phase0.spectral_profiler import compute_laplacian_spectrum, partition_into_bands
from .noise_shaper import ImportanceWeights, shape_noise, shape_noise_with_temporal_ramp
from .score_network import SimpleScoreNetwork
from .sde import VPSDE, CosineScheduleSDE

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training configuration."""

    n_epochs: int = 500
    lr: float = 1e-3
    batch_timesteps: int = 16
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cpu"
    use_spectral_noise: bool = False
    t_knee: float | None = None
    alpha: float = 1.0
    epsilon: float = 1e-3
    B: int = 8
    hidden_dim: int = 128
    n_layers: int = 3
    # SDE params
    beta_min: float = 0.1
    beta_max: float = 10.0
    sde_type: str = "vpsde"  # "vpsde" or "cosine"
    n_gen_steps: int = 200
    # Alt-2 additions
    use_ema: bool = False
    ema_decay: float = 0.999
    use_lr_scheduler: bool = False


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {name: p.clone().detach() for name, p in model.named_parameters()}

    def update(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Apply EMA weights, return original weights for restoration."""
        original = {}
        for name, p in model.named_parameters():
            original[name] = p.data.clone()
            p.data.copy_(self.shadow[name])
        return original

    def restore(self, model: nn.Module, original: dict[str, torch.Tensor]) -> None:
        """Restore original weights after EMA evaluation."""
        for name, p in model.named_parameters():
            p.data.copy_(original[name])


class Trainer:
    """Train a score network with denoising score matching.

    Loss = E_t E_{x_0} E_{noise} [ ||score_net(x_t, t) - (-noise / std(t))||^2 ]

    When use_spectral_noise=True, noise is shaped via the FANS mechanism.
    """

    def __init__(
        self,
        config: TrainConfig,
        graph: nx.Graph,
        features: np.ndarray,
        importance_weights: ImportanceWeights | None = None,
    ):
        self.config = config
        self.device = torch.device(config.device)

        # Graph structure
        self.graph = graph
        edges = list(graph.edges())
        edge_list = edges + [(v, u) for u, v in edges]
        src, dst = zip(*edge_list) if edge_list else ([], [])
        self.edge_index = torch.tensor([list(src), list(dst)], dtype=torch.long, device=self.device)

        # Features
        self.features = torch.tensor(features, dtype=torch.float32, device=self.device)
        self.n_nodes, self.n_features = features.shape

        # Spectral decomposition
        eigenvalues, eigenvectors = compute_laplacian_spectrum(graph)
        self.eigenvalues = eigenvalues
        self.eigenvectors_np = eigenvectors
        self.eigenvectors = torch.tensor(eigenvectors, dtype=torch.float32, device=self.device)
        _, self.band_indices = partition_into_bands(eigenvalues, B=config.B)

        self.importance_weights = importance_weights

        # SDE
        if config.sde_type == "cosine":
            self.sde = CosineScheduleSDE()
        else:
            self.sde = VPSDE(beta_min=config.beta_min, beta_max=config.beta_max)

        # Model
        self.model = SimpleScoreNetwork(
            n_features=self.n_features,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        # LR scheduler: cosine annealing
        self.scheduler = None
        if config.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.n_epochs, eta_min=config.lr * 0.01
            )

        # EMA
        self.ema = None
        if config.use_ema:
            self.ema = EMA(self.model, decay=config.ema_decay)

    def _get_noise(self, t: float) -> torch.Tensor:
        """Generate noise, optionally shaped spectrally."""
        noise = torch.randn(self.n_nodes, self.n_features, device=self.device)

        if not self.config.use_spectral_noise or self.importance_weights is None:
            return noise

        if self.config.t_knee is not None:
            return shape_noise_with_temporal_ramp(
                noise, self.eigenvectors, self.band_indices,
                self.importance_weights, t, self.config.t_knee,
            )
        else:
            return shape_noise(
                noise, self.eigenvectors, self.band_indices,
                self.importance_weights,
            )

    def train(self) -> dict[str, list[float]]:
        """Run training. Returns dict with 'loss' key containing per-epoch losses."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        self.model.train()
        history: dict[str, list[float]] = {"loss": []}

        for epoch in range(self.config.n_epochs):
            epoch_loss = 0.0

            ts = np.random.uniform(1e-5, self.sde.T, size=self.config.batch_timesteps)

            for t in ts:
                noise = self._get_noise(t)
                x_t, _ = self.sde.perturb(self.features, t, noise=noise)

                _, std = self.sde.marginal_params(t)
                if std > 1e-8:
                    target = -noise / std
                else:
                    target = torch.zeros_like(noise)

                t_tensor = torch.tensor([t], dtype=torch.float32, device=self.device)
                score_pred = self.model(x_t, t_tensor, self.edge_index)

                loss = nn.functional.mse_loss(score_pred, target)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

                # EMA update
                if self.ema is not None:
                    self.ema.update(self.model)

                epoch_loss += loss.item()

            if self.scheduler is not None:
                self.scheduler.step()

            avg_loss = epoch_loss / self.config.batch_timesteps
            history["loss"].append(avg_loss)

            if (epoch + 1) % 100 == 0 or epoch == 0:
                logger.info(f"  Epoch {epoch + 1}/{self.config.n_epochs}, loss={avg_loss:.6f}")

        return history

    @torch.no_grad()
    def generate(self, n_steps: int | None = None) -> np.ndarray:
        """Generate features via reverse SDE sampling.

        Uses EMA weights if available.
        """
        if n_steps is None:
            n_steps = self.config.n_gen_steps

        self.model.eval()

        # Apply EMA weights for generation if available
        original_weights = None
        if self.ema is not None:
            original_weights = self.ema.apply(self.model)

        x = torch.randn(self.n_nodes, self.n_features, device=self.device)

        dt = -self.sde.T / n_steps
        ts = np.linspace(self.sde.T, 1e-5, n_steps)

        for t in ts:
            t_tensor = torch.tensor([t], dtype=torch.float32, device=self.device)
            score = self.model(x, t_tensor, self.edge_index)
            x = self.sde.reverse_step(x, score, t, dt)

        # Restore original weights
        if self.ema is not None and original_weights is not None:
            self.ema.restore(self.model, original_weights)

        self.model.train()
        return x.cpu().numpy()
