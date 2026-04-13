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
from .noise_shaper import (
    ImportanceWeights,
    ModeImportanceWeights,
    shape_noise,
    shape_noise_per_mode,
    shape_noise_with_temporal_ramp,
)
from .score_network import SimpleScoreNetwork
from .sde import VPSDE, CosineScheduleSDE
from .spectral_loss import spectral_fidelity_loss, tweedie_denoise

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training configuration."""

    n_epochs: int = 2000
    lr: float = 1e-3
    batch_timesteps: int = 32
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cpu"
    use_spectral_noise: bool = False
    noise_shaping: str = "uniform"  # "uniform", "band", or "mode"
    t_knee: float | None = None  # DEPRECATED (H2 removed): no longer used by the main pipeline
    alpha: float = 1.0
    epsilon: float = 1e-3
    B: int = 8
    hidden_dim: int = 128
    n_layers: int = 3
    # SDE params
    beta_min: float = 0.1
    beta_max: float = 10.0
    sde_type: str = "cosine"  # default to cosine (proven stable)
    n_gen_steps: int = 200
    # Stability
    use_ema: bool = True  # default on (proven helpful)
    ema_decay: float = 0.999
    use_lr_scheduler: bool = True  # default on
    # Spectral loss
    use_spectral_loss: bool = False
    spectral_loss_weight: float = 0.1
    # Dataset size
    n_train_samples: int = 500  # number of feature realizations per graph
    # Architecture
    conv_type: str = "gcn"  # "gcn" or "transformer"
    # NS-A: log-SNR uniform timestep sampling
    t_sampling: str = "uniform"  # "uniform" or "log_snr"
    # NS-C: min-SNR-γ loss weighting (None = disabled, 5.0 = standard)
    min_snr_gamma: float | None = None


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {name: p.clone().detach() for name, p in model.named_parameters()}

    def update(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model: nn.Module) -> dict[str, torch.Tensor]:
        original = {}
        for name, p in model.named_parameters():
            original[name] = p.data.clone()
            p.data.copy_(self.shadow[name])
        return original

    def restore(self, model: nn.Module, original: dict[str, torch.Tensor]) -> None:
        for name, p in model.named_parameters():
            p.data.copy_(original[name])


class Trainer:
    """Train a score network with denoising score matching on a feature dataset.

    Key fix: trains on a DATASET of N feature realizations per graph,
    not a single feature matrix. Each training step samples a random
    feature realization and a random timestep.
    """

    def __init__(
        self,
        config: TrainConfig,
        graph: nx.Graph,
        feature_dataset: np.ndarray,  # [N, n_nodes, n_features]
        importance_weights: ImportanceWeights | None = None,
        mode_weights: ModeImportanceWeights | None = None,
    ):
        self.config = config
        self.device = torch.device(config.device)

        # Graph structure
        self.graph = graph
        edges = list(graph.edges())
        edge_list = edges + [(v, u) for u, v in edges]
        src, dst = zip(*edge_list) if edge_list else ([], [])
        self.edge_index = torch.tensor([list(src), list(dst)], dtype=torch.long, device=self.device)

        # Feature dataset: [N, n_nodes, n_features]
        self.dataset = torch.tensor(feature_dataset, dtype=torch.float32, device=self.device)
        self.n_samples, self.n_nodes, self.n_features = self.dataset.shape
        logger.info(f"  Dataset: {self.n_samples} samples, {self.n_nodes} nodes, {self.n_features} features")

        # Spectral decomposition
        eigenvalues, eigenvectors = compute_laplacian_spectrum(graph)
        self.eigenvalues = eigenvalues
        self.eigenvectors_np = eigenvectors
        self.eigenvectors = torch.tensor(eigenvectors, dtype=torch.float32, device=self.device)
        _, self.band_indices = partition_into_bands(eigenvalues, B=config.B)

        self.importance_weights = importance_weights
        self.mode_weights = mode_weights

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
            conv_type=config.conv_type,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        self.scheduler = None
        if config.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.n_epochs, eta_min=config.lr * 0.01
            )

        self.ema = None
        if config.use_ema:
            self.ema = EMA(self.model, decay=config.ema_decay)

    def _snr_to_t(self, target_snr: float) -> float:
        """Binary search to find t such that SNR(t) ≈ target_snr.

        SNR(t) = alpha_bar(t) / (1 - alpha_bar(t)).
        alpha_bar is monotonically decreasing, so higher t → lower SNR.

        Args:
            target_snr: The target signal-to-noise ratio (must be positive).

        Returns:
            Timestep t in [1e-5, T] such that SNR(t) ≈ target_snr.
        """
        lo, hi = 1e-5, self.sde.T
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            ab = self.sde.alpha_bar(mid)
            snr = ab / max(1.0 - ab, 1e-10)
            if snr > target_snr:
                lo = mid  # higher t needed to reduce SNR
            else:
                hi = mid
            if hi - lo < 1e-6:
                break
        return 0.5 * (lo + hi)

    def _get_noise(self, t: float) -> torch.Tensor:
        """Generate noise, optionally shaped spectrally.

        Resolves effective shaping mode from noise_shaping config and legacy
        use_spectral_noise flag. Priority: noise_shaping > use_spectral_noise.
        """
        noise = torch.randn(self.n_nodes, self.n_features, device=self.device)

        # Resolve effective shaping mode
        effective = self.config.noise_shaping
        if effective == "uniform" and self.config.use_spectral_noise:
            effective = "band"  # legacy compatibility

        if effective == "mode" and self.mode_weights is not None:
            return shape_noise_per_mode(
                noise, self.eigenvectors, self.mode_weights,
            )

        if effective == "band" and self.importance_weights is not None:
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

        return noise

    def train(self) -> dict[str, list[float]]:
        """Run training, sampling random features + random timesteps each step."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        self.model.train()
        history: dict[str, list[float]] = {"loss": []}

        for epoch in range(self.config.n_epochs):
            epoch_loss = 0.0
            n_steps = 0

            # Each epoch: sample batch_timesteps steps, each with a random sample + random t
            if self.config.t_sampling == "log_snr":
                # NS-A: uniform sampling in log-SNR space
                ab_T = self.sde.alpha_bar(self.sde.T)
                ab_0 = self.sde.alpha_bar(1e-5)
                log_snr_min = np.log(ab_T / max(1.0 - ab_T, 1e-10))
                log_snr_max = np.log(ab_0 / max(1.0 - ab_0, 1e-10))
                log_snrs = np.random.uniform(log_snr_min, log_snr_max, size=self.config.batch_timesteps)
                ts = np.array([self._snr_to_t(np.exp(ls)) for ls in log_snrs])
            else:
                ts = np.random.uniform(1e-5, self.sde.T, size=self.config.batch_timesteps)
            sample_indices = np.random.randint(0, self.n_samples, size=self.config.batch_timesteps)

            for t, idx in zip(ts, sample_indices):
                x_0 = self.dataset[idx]  # [n_nodes, n_features] — random sample from dataset

                noise = self._get_noise(t)
                x_t, _ = self.sde.perturb(x_0, t, noise=noise)

                # ε-prediction: target is the noise itself
                target = noise

                t_tensor = torch.tensor([t], dtype=torch.float32, device=self.device)
                eps_pred = self.model(x_t, t_tensor, self.edge_index)

                loss = nn.functional.mse_loss(eps_pred, target)

                # NS-C: min-SNR-γ loss weighting
                if self.config.min_snr_gamma is not None:
                    mean_coeff, std = self.sde.marginal_params(t)
                    snr = mean_coeff**2 / max(std**2, 1e-8)
                    weight = min(snr, self.config.min_snr_gamma) / snr
                    loss = weight * loss

                # Spectral loss
                if self.config.use_spectral_loss:
                    mean_coeff, std = self.sde.marginal_params(t)
                    if std > 1e-4:
                        x_hat_0 = tweedie_denoise(x_t, eps_pred, mean_coeff, std)
                        spec_loss = spectral_fidelity_loss(
                            x_hat_0, x_0,
                            self.eigenvectors, self.band_indices,
                            self.importance_weights,
                        )
                        loss = loss + self.config.spectral_loss_weight * spec_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

                if self.ema is not None:
                    self.ema.update(self.model)

                epoch_loss += loss.item()
                n_steps += 1

            if self.scheduler is not None:
                self.scheduler.step()

            avg_loss = epoch_loss / max(n_steps, 1)
            history["loss"].append(avg_loss)

            if (epoch + 1) % 100 == 0 or epoch == 0:
                logger.info(f"  Epoch {epoch + 1}/{self.config.n_epochs}, loss={avg_loss:.6f}")

        return history

    @torch.no_grad()
    def generate(self, n_steps: int | None = None, n_samples: int = 1) -> np.ndarray:
        """Generate feature samples via deterministic DDIM reverse process.

        Args:
            n_steps: Number of DDIM steps.
            n_samples: Number of independent samples to generate.

        Returns:
            Generated features [n_samples, n_nodes, n_features].
        """
        if n_steps is None:
            n_steps = self.config.n_gen_steps

        self.model.eval()
        original_weights = None
        if self.ema is not None:
            original_weights = self.ema.apply(self.model)

        # Time schedule: T → 0
        ts = np.linspace(self.sde.T, 0, n_steps + 1)  # includes t=0

        all_samples = []
        for _ in range(n_samples):
            x = torch.randn(self.n_nodes, self.n_features, device=self.device)

            for i in range(n_steps):
                t_now = float(ts[i])
                t_next = float(ts[i + 1])
                t_tensor = torch.tensor([t_now], dtype=torch.float32, device=self.device)
                eps_pred = self.model(x, t_tensor, self.edge_index)
                x = self.sde.ddim_step(x, eps_pred, t_now, t_next)

            all_samples.append(x.cpu().numpy())

        if self.ema is not None and original_weights is not None:
            self.ema.restore(self.model, original_weights)

        self.model.train()
        return np.stack(all_samples)  # [n_samples, n_nodes, n_features]

    @torch.no_grad()
    def sanity_check(self, train_data: np.ndarray, n_gen: int = 5) -> dict:
        """Post-training sanity check: compare generated vs training statistics.

        Args:
            train_data: Training dataset [N, n_nodes, n_features].
            n_gen: Number of samples to generate for comparison.

        Returns:
            Dict with 'train_std', 'gen_std', 'std_ratio', 'warnings'.
        """
        from graph_fans.phase0.spectral_profiler import compute_band_energy

        gen_data = self.generate(n_samples=n_gen)

        train_std = float(train_data.std())
        gen_std = float(gen_data.std())
        std_ratio = gen_std / max(train_std, 1e-8)

        # Spectral profile comparison
        def mean_profile(data):
            profiles = []
            for features in data:
                energy = compute_band_energy(features, self.eigenvectors_np, self.band_indices)
                total = energy.sum()
                profiles.append(energy / total if total > 0 else energy)
            return np.stack(profiles).mean(axis=0)

        train_profile = mean_profile(train_data[:min(50, len(train_data))])
        gen_profile = mean_profile(gen_data)
        spectral_l2 = float(np.linalg.norm(train_profile - gen_profile))

        warnings = []
        if std_ratio > 3.0:
            warnings.append(f"Generated std {gen_std:.2f} is {std_ratio:.1f}× training std {train_std:.2f}")
        if std_ratio < 0.1:
            warnings.append(f"Generated std {gen_std:.4f} is only {std_ratio:.2f}× training std {train_std:.2f}")
        if spectral_l2 > 0.5:
            warnings.append(f"Spectral L2 distance {spectral_l2:.3f} > 0.5 — generated profile differs significantly")

        for w in warnings:
            logger.warning(f"  SANITY CHECK: {w}")

        if not warnings:
            logger.info(f"  Sanity check passed: gen_std={gen_std:.3f}, train_std={train_std:.3f}, spectral_L2={spectral_l2:.3f}")

        return {
            "train_std": train_std,
            "gen_std": gen_std,
            "std_ratio": std_ratio,
            "spectral_l2": spectral_l2,
            "train_profile": train_profile.tolist(),
            "gen_profile": gen_profile.tolist(),
            "warnings": warnings,
        }
