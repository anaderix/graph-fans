"""Per-mode diffusion schedule: mode-specific t_max derived from spectral energy.

High-energy modes (low frequency, community structure) need more diffusion time
to corrupt. Low-energy modes (high frequency, close to noise) need less.
Reuses CosineScheduleSDE for all actual SDE math.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from graph_fans.phase2.sde import CosineScheduleSDE

logger = logging.getLogger(__name__)


@dataclass
class ModeSchedule:
    """Schedule parameters for a single eigenmode."""

    mode_idx: int
    eigenvalue: float
    energy: float
    t_max: float


class ModeScheduleSet:
    """Per-mode schedule collection wrapping CosineScheduleSDE.

    High-energy modes (low frequency) get larger t_max (more noise needed
    to corrupt them). Low-energy modes (high frequency) get smaller t_max
    (already close to noise, less corruption needed).

    t_max(k) = T * (E_k / E_max)^energy_exponent, clamped [t_max_floor * T, T].
    """

    def __init__(
        self,
        eigenvalues: np.ndarray,
        mode_energies: np.ndarray,
        base_sde: CosineScheduleSDE,
        energy_exponent: float = 0.5,
        t_max_floor: float = 0.1,
    ):
        T = base_sde.T
        E_max = float(mode_energies.max()) if mode_energies.max() > 0 else 1.0
        self.base_sde = base_sde
        self.schedules: list[ModeSchedule] = []

        for k in range(len(eigenvalues)):
            ratio = float(mode_energies[k] / E_max) if E_max > 0 else 1.0
            t_max_k = T * ratio ** energy_exponent
            t_max_k = float(np.clip(t_max_k, t_max_floor * T, T))
            self.schedules.append(ModeSchedule(
                mode_idx=k,
                eigenvalue=float(eigenvalues[k]),
                energy=float(mode_energies[k]),
                t_max=t_max_k,
            ))

        logger.debug(
            "ModeScheduleSet: %d modes, t_max range [%.3f, %.3f]",
            len(self.schedules),
            min(s.t_max for s in self.schedules),
            max(s.t_max for s in self.schedules),
        )

    @property
    def n_modes(self) -> int:
        return len(self.schedules)

    def sample_t(self, mode_idx: int, rng: np.random.Generator) -> float:
        """Sample uniform t in [eps, t_max(k)]."""
        eps = 1e-5
        t_max = self.schedules[mode_idx].t_max
        return float(rng.uniform(eps, t_max))

    def get_ddim_grid(self, mode_idx: int, n_steps: int) -> np.ndarray:
        """Return DDIM timestep grid from t_max(k) to ~0.

        Returns array of shape [n_steps + 1], decreasing from t_max(k) to eps.
        """
        t_max = self.schedules[mode_idx].t_max
        return np.linspace(t_max, 1e-5, n_steps + 1)

    def alpha_bar(self, mode_idx: int, t: float) -> float:
        """Delegate to base SDE alpha_bar (schedule is shared, only t_max differs)."""
        return self.base_sde.alpha_bar(t)

    def perturb(
        self,
        x_0: torch.Tensor,
        mode_idx: int,
        t: float,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffuse using mode-specific time. Delegates to base SDE."""
        return self.base_sde.perturb(x_0, t, noise=noise)

    def ddim_step(
        self,
        x_t: torch.Tensor,
        eps_pred: torch.Tensor,
        mode_idx: int,
        t_now: float,
        t_next: float,
    ) -> torch.Tensor:
        """DDIM deterministic step using base SDE alpha_bar."""
        return self.base_sde.ddim_step(x_t, eps_pred, t_now, t_next)
