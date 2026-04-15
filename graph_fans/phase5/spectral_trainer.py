"""Training loop and generation for independent per-mode spectral diffusion.

Each eigenmode k is treated as an independent d-dimensional denoising problem.
A shared MLP conditioned on (lambda_k, t, E_k) learns to predict noise.
All modes are batched into a single forward pass for GPU efficiency.

Key difference from Phase 3a (NO-GO): Phase 3a shaped noise in a spatial GCN
pipeline, creating train/generate mismatch. Phase 5a diffuses entirely in the
spectral domain -- no GCN, no mismatch.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from graph_fans.phase0.spectral_profiler import compute_laplacian_spectrum
from graph_fans.phase2.sde import CosineScheduleSDE

from .mode_schedule import ModeScheduleSet
from .spectral_score_network import SpectralScoreNetwork

logger = logging.getLogger(__name__)


@dataclass
class SpectralTrainConfig:
    """Configuration for spectral per-mode diffusion training."""

    n_epochs: int = 500
    lr: float = 1e-3
    hidden_dim: int = 128
    n_layers: int = 3
    n_features: int = 4
    energy_exponent: float = 0.5
    t_max_floor: float = 0.1
    n_gen_steps: int = 200
    use_ema: bool = True
    ema_decay: float = 0.999
    use_lr_scheduler: bool = True
    seed: int = 42
    device: str = "cpu"
    grad_clip: float = 1.0
    weight_decay: float = 1e-4


def _ema_update(
    ema_model: nn.Module,
    model: nn.Module,
    decay: float,
) -> None:
    """Update EMA model parameters."""
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


class SpectralTrainer:
    """Train a shared MLP for independent per-mode spectral diffusion.

    Pipeline:
    1. Eigendecompose graph Laplacian (once, amortized).
    2. Project all training features to spectral domain [N, n_modes, d].
    3. Compute per-mode energy from training data only.
    4. Train: sample a realization, batch all modes, epsilon-prediction + MSE.
    5. Generate: per-mode DDIM from t_max(k) to 0, reconstruct x = U @ c.
    """

    def __init__(
        self,
        graph: nx.Graph,
        features_list: np.ndarray,  # [N_train, n_nodes, n_features]
        config: SpectralTrainConfig,
    ):
        self.config = config
        self.device = torch.device(config.device)

        # Eigendecomposition (amortized, computed once)
        eigenvalues, eigenvectors = compute_laplacian_spectrum(graph)
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors  # [n_nodes, n_modes]
        self.n_modes = eigenvectors.shape[1]

        # Project all training features to spectral domain: [N, n_modes, d]
        U_T = eigenvectors.T  # [n_modes, n_nodes]
        self.spectral_data = np.stack([
            U_T @ feat for feat in features_list
        ])  # [N_train, n_modes, n_features]
        logger.info(
            "  Spectral data: %d samples, %d modes, %d features",
            self.spectral_data.shape[0], self.n_modes, config.n_features,
        )

        # Compute per-mode energy from training data only (no data leakage)
        self.mode_energies = (self.spectral_data ** 2).mean(axis=(0, 2))  # [n_modes]

        # Build per-mode schedule set
        base_sde = CosineScheduleSDE()
        self.base_sde = base_sde
        self.schedule_set = ModeScheduleSet(
            eigenvalues,
            self.mode_energies,
            base_sde,
            energy_exponent=config.energy_exponent,
            t_max_floor=config.t_max_floor,
        )

        # Build shared MLP
        self.model = SpectralScoreNetwork(
            n_features=config.n_features,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # EMA
        self.ema_model: nn.Module | None = None
        if config.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # LR scheduler
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        if config.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.n_epochs, eta_min=config.lr * 0.01,
            )

        # Precompute conditioning tensors (constant across training)
        self._lambda_k = torch.tensor(
            self.eigenvalues, dtype=torch.float32, device=self.device,
        ).unsqueeze(-1)  # [n_modes, 1]
        self._E_k = torch.tensor(
            self.mode_energies, dtype=torch.float32, device=self.device,
        ).unsqueeze(-1)  # [n_modes, 1]

    def train(self) -> dict:
        """Training loop: per epoch, sample one realization, batch all modes.

        Returns:
            Dict with 'losses' (all epoch losses) and 'final_loss' (mean of last 50).
        """
        rng = np.random.default_rng(self.config.seed)
        torch.manual_seed(self.config.seed)
        losses: list[float] = []
        self.model.train()
        t_start = time.time()

        for epoch in range(self.config.n_epochs):
            # Sample random training feature realization
            idx = int(rng.integers(len(self.spectral_data)))
            c_all = self.spectral_data[idx]  # [n_modes, d]

            # Sample t_k per mode (each mode has its own t_max)
            t_per_mode = np.array([
                self.schedule_set.sample_t(k, rng) for k in range(self.n_modes)
            ])

            # Convert to tensors
            c_0 = torch.tensor(c_all, dtype=torch.float32, device=self.device)
            noise = torch.randn_like(c_0)

            # Vectorized alpha_bar for each mode's t_k
            alpha_bars = torch.tensor(
                [self.base_sde.alpha_bar(float(t_per_mode[k])) for k in range(self.n_modes)],
                dtype=torch.float32, device=self.device,
            ).unsqueeze(-1)  # [n_modes, 1]

            # Forward diffusion: c_t = sqrt(alpha_bar) * c_0 + sqrt(1 - alpha_bar) * noise
            c_t = torch.sqrt(alpha_bars) * c_0 + torch.sqrt(1.0 - alpha_bars) * noise

            # Time conditioning
            t_cond = torch.tensor(
                t_per_mode, dtype=torch.float32, device=self.device,
            ).unsqueeze(-1)  # [n_modes, 1]

            # Forward pass (batched over all modes -- single MLP call)
            eps_pred = self.model(c_t, self._lambda_k, t_cond, self._E_k)

            # MSE loss averaged over all modes
            loss = ((eps_pred - noise) ** 2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            if self.ema_model is not None:
                _ema_update(self.ema_model, self.model, self.config.ema_decay)
            if self.scheduler is not None:
                self.scheduler.step()

            losses.append(loss.item())

            if (epoch + 1) % 100 == 0 or epoch == 0:
                logger.info(
                    "  Epoch %d/%d, loss=%.6f",
                    epoch + 1, self.config.n_epochs, loss.item(),
                )

        train_time = time.time() - t_start
        final_loss = float(np.mean(losses[-50:])) if len(losses) >= 50 else float(np.mean(losses))
        logger.info("  Training complete: %.1fs, final_loss=%.6f", train_time, final_loss)

        return {
            "losses": losses,
            "final_loss": final_loss,
            "train_time": train_time,
        }

    @torch.no_grad()
    def generate(
        self,
        n_steps: int | None = None,
        n_samples: int = 1,
    ) -> list[np.ndarray]:
        """Generate features via per-mode DDIM, reconstruct to spatial domain.

        Each mode follows its own DDIM grid (t_max(k) to 0). All modes are
        batched into a single MLP forward pass per DDIM step.

        Args:
            n_steps: Number of DDIM steps per mode.
            n_samples: Number of independent samples to generate.

        Returns:
            List of [n_nodes, n_features] arrays.
        """
        if n_steps is None:
            n_steps = self.config.n_gen_steps

        model = self.ema_model if self.ema_model is not None else self.model
        model.eval()

        # Precompute DDIM grids for all modes
        ddim_grids = [
            self.schedule_set.get_ddim_grid(k, n_steps) for k in range(self.n_modes)
        ]

        results: list[np.ndarray] = []
        for _ in range(n_samples):
            # Initialize: c_k(T_k) ~ N(0, I_d) for each mode
            c_gen = torch.randn(
                self.n_modes, self.config.n_features, device=self.device,
            )

            # Scale initial noise: at t_max(k), signal is sqrt(1 - alpha_bar(t_max(k))) * noise
            for k in range(self.n_modes):
                alpha_bar_start = self.schedule_set.alpha_bar(k, ddim_grids[k][0])
                c_gen[k] = c_gen[k] * np.sqrt(1.0 - alpha_bar_start)

            # DDIM loop: iterate over timestep indices (all modes advance together)
            for step_idx in range(n_steps):
                # Gather t_now and t_next for each mode
                t_now_per_mode = np.array([ddim_grids[k][step_idx] for k in range(self.n_modes)])
                t_next_per_mode = np.array([ddim_grids[k][step_idx + 1] for k in range(self.n_modes)])

                t_cond = torch.tensor(
                    t_now_per_mode, dtype=torch.float32, device=self.device,
                ).unsqueeze(-1)  # [n_modes, 1]

                # Batched MLP forward pass
                eps_pred = model(c_gen, self._lambda_k, t_cond, self._E_k)

                # Vectorized DDIM step for all modes
                ab_now = torch.tensor(
                    [self.base_sde.alpha_bar(float(t_now_per_mode[k])) for k in range(self.n_modes)],
                    dtype=torch.float32, device=self.device,
                ).unsqueeze(-1)  # [n_modes, 1]

                # x_0_hat = (c_t - sqrt(1 - ab) * eps) / sqrt(ab)
                x_0_hat = (c_gen - torch.sqrt(1.0 - ab_now) * eps_pred) / torch.sqrt(ab_now.clamp(min=1e-8))

                # Check if we're at final step
                ab_next = torch.tensor(
                    [self.base_sde.alpha_bar(float(t_next_per_mode[k])) for k in range(self.n_modes)],
                    dtype=torch.float32, device=self.device,
                ).unsqueeze(-1)  # [n_modes, 1]

                # For final step (t_next ~ 0), use x_0_hat; otherwise interpolate
                c_gen = torch.sqrt(ab_next) * x_0_hat + torch.sqrt(1.0 - ab_next) * eps_pred

            # Reconstruct: x = U @ c
            c_np = c_gen.cpu().numpy()  # [n_modes, n_features]
            x_gen = self.eigenvectors @ c_np  # [n_nodes, n_features]
            results.append(x_gen)

        return results

    @torch.no_grad()
    def sanity_check(
        self,
        ref_features: np.ndarray,
        gen_features: np.ndarray,
    ) -> dict:
        """Quick sanity check: std ratio and spectral profile distance.

        Args:
            ref_features: [n_nodes, n_features] single reference sample.
            gen_features: [n_nodes, n_features] single generated sample.

        Returns:
            Dict with 'std_ratio', 'spectral_l2'.
        """
        ref_std = float(ref_features.std())
        gen_std = float(gen_features.std())
        std_ratio = gen_std / max(ref_std, 1e-8)

        # Spectral energy profile comparison
        spectral_gen = self.eigenvectors.T @ gen_features  # [n_modes, d]
        spectral_ref = self.eigenvectors.T @ ref_features  # [n_modes, d]
        gen_energy = (spectral_gen ** 2).mean(axis=1)
        ref_energy = (spectral_ref ** 2).mean(axis=1)
        spectral_l2 = float(np.sqrt(((gen_energy - ref_energy) ** 2).mean()))

        warnings = []
        if std_ratio > 3.0:
            warnings.append(f"Generated std {gen_std:.2f} is {std_ratio:.1f}x ref std {ref_std:.2f}")
        if std_ratio < 0.1:
            warnings.append(f"Generated std {gen_std:.4f} is only {std_ratio:.2f}x ref std {ref_std:.2f}")

        for w in warnings:
            logger.warning("  SANITY CHECK: %s", w)

        return {
            "std_ratio": float(std_ratio),
            "spectral_l2": float(spectral_l2),
            "warnings": warnings,
        }
