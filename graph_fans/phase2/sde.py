"""Variance-Preserving SDE for node feature diffusion."""

from __future__ import annotations

import numpy as np
import torch


class VPSDE:
    """Variance-Preserving SDE with linear beta schedule.

    dx = -0.5 * beta(t) * x * dt + sqrt(beta(t)) * dw

    The marginal distribution q(x_t | x_0) is Gaussian:
        x_t = mean_coeff(t) * x_0 + std(t) * noise
    where:
        mean_coeff(t) = exp(-0.5 * integral_0^t beta(s) ds)
        std(t) = sqrt(1 - mean_coeff(t)^2)
    """

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        T: float = 1.0,
        n_timesteps: int = 1000,
    ):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self.n_timesteps = n_timesteps

    def beta(self, t: float) -> float:
        """Linear beta schedule."""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def _integral_beta(self, t: float) -> float:
        """Integral of beta(s) from 0 to t."""
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2

    def marginal_params(self, t: float) -> tuple[float, float]:
        """Return (mean_coeff, std) for q(x_t | x_0).

        Args:
            t: Timestep in [0, T].

        Returns:
            (mean_coeff, std) such that x_t = mean_coeff * x_0 + std * noise.
        """
        log_mean_coeff = -0.5 * self._integral_beta(t)
        mean_coeff = np.exp(log_mean_coeff)
        std = np.sqrt(max(1.0 - mean_coeff**2, 0.0))
        return float(mean_coeff), float(std)

    def perturb(
        self,
        x_0: torch.Tensor,
        t: float,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: sample x_t given x_0.

        Args:
            x_0: Clean data [n_nodes, n_features].
            t: Timestep.
            noise: Optional pre-generated noise (for spectral shaping).

        Returns:
            (x_t, noise_used).
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        mean_coeff, std = self.marginal_params(t)
        x_t = mean_coeff * x_0 + std * noise
        return x_t, noise

    def reverse_step(
        self,
        x_t: torch.Tensor,
        score: torch.Tensor,
        t: float,
        dt: float,
    ) -> torch.Tensor:
        """One reverse SDE step using predicted score.

        Euler-Maruyama discretization of the reverse SDE:
        dx = [-0.5 * beta(t) * x - beta(t) * score] * dt + sqrt(beta(t)) * dw_reverse

        Args:
            x_t: Current state [n_nodes, n_features].
            score: Predicted score [n_nodes, n_features].
            t: Current timestep.
            dt: Step size (negative for reverse).

        Returns:
            x_{t+dt} (one step back in time).
        """
        beta_t = self.beta(t)
        drift = -0.5 * beta_t * x_t - beta_t * score
        diffusion = np.sqrt(beta_t)

        noise = torch.randn_like(x_t)
        # dt is negative (going backward), so we use abs(dt) for the noise term
        x_prev = x_t + drift * dt + diffusion * np.sqrt(abs(dt)) * noise
        return x_prev
