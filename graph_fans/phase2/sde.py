"""SDE implementations for node feature diffusion."""

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
        """Return (mean_coeff, std) for q(x_t | x_0)."""
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
        """Forward diffusion: sample x_t given x_0."""
        if noise is None:
            noise = torch.randn_like(x_0)
        mean_coeff, std = self.marginal_params(t)
        x_t = mean_coeff * x_0 + std * noise
        return x_t, noise

    def alpha_bar(self, t: float) -> float:
        """Cumulative signal retention: alpha_bar(t) = mean_coeff(t)^2."""
        mean_coeff, _ = self.marginal_params(t)
        return float(mean_coeff ** 2)

    def ddim_step(
        self,
        x_t: torch.Tensor,
        eps_pred: torch.Tensor,
        t_now: float,
        t_next: float,
    ) -> torch.Tensor:
        """Deterministic DDIM reverse step (eta=0).

        Args:
            x_t: Current noisy data.
            eps_pred: Predicted noise (epsilon).
            t_now: Current timestep.
            t_next: Next timestep (smaller). If <= 0, returns Tweedie estimate.

        Returns:
            Denoised data at t_next.
        """
        ab_now = max(self.alpha_bar(t_now), 1e-8)
        x_0_hat = (x_t - np.sqrt(1 - ab_now) * eps_pred) / np.sqrt(ab_now)
        if t_next <= 0:
            return x_0_hat
        ab_next = max(self.alpha_bar(t_next), 1e-8)
        return np.sqrt(ab_next) * x_0_hat + np.sqrt(1 - ab_next) * eps_pred

    def reverse_step(
        self,
        x_t: torch.Tensor,
        score: torch.Tensor,
        t: float,
        dt: float,
    ) -> torch.Tensor:
        """One reverse SDE step (Euler-Maruyama). DEPRECATED: use ddim_step."""
        beta_t = self.beta(t)
        drift = -0.5 * beta_t * x_t - beta_t * score
        diffusion = np.sqrt(beta_t)
        noise = torch.randn_like(x_t)
        x_prev = x_t + drift * dt + diffusion * np.sqrt(abs(dt)) * noise
        return x_prev


class CosineScheduleSDE:
    """Diffusion SDE with cosine noise schedule (Nichol & Dhariwal, 2021).

    Uses alpha_bar(t) = cos^2(pi/2 * (t + s) / (1 + s)) with offset s=0.008.
    This gives a much gentler noise increase than linear beta, which is critical
    for stability on small graph signals.

    The marginal is:
        x_t = sqrt(alpha_bar(t)) * x_0 + sqrt(1 - alpha_bar(t)) * noise
    """

    def __init__(self, T: float = 1.0, s: float = 0.008, n_timesteps: int = 1000):
        self.T = T
        self.s = s
        self.n_timesteps = n_timesteps

    def _alpha_bar(self, t: float) -> float:
        """Cosine schedule: alpha_bar(t) = cos^2(...)."""
        return np.cos(np.pi / 2 * (t / self.T + self.s) / (1 + self.s)) ** 2

    def _alpha_bar_clipped(self, t: float) -> float:
        """Clipped to prevent numerical issues near t=T."""
        return max(self._alpha_bar(t), 1e-5)

    def beta(self, t: float) -> float:
        """Instantaneous beta derived from alpha_bar schedule."""
        dt = 1e-4
        ab_t = self._alpha_bar_clipped(t)
        ab_t_dt = self._alpha_bar_clipped(t + dt)
        # beta(t) = -d/dt log(alpha_bar(t))
        beta = -(np.log(ab_t_dt) - np.log(ab_t)) / dt
        return max(float(beta), 0.0)

    def marginal_params(self, t: float) -> tuple[float, float]:
        """Return (mean_coeff, std) for q(x_t | x_0)."""
        ab = self._alpha_bar_clipped(t)
        mean_coeff = np.sqrt(ab)
        std = np.sqrt(1.0 - ab)
        return float(mean_coeff), float(std)

    def perturb(
        self,
        x_0: torch.Tensor,
        t: float,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: sample x_t given x_0."""
        if noise is None:
            noise = torch.randn_like(x_0)
        mean_coeff, std = self.marginal_params(t)
        x_t = mean_coeff * x_0 + std * noise
        return x_t, noise

    def alpha_bar(self, t: float) -> float:
        """Cumulative signal retention (same as _alpha_bar_clipped)."""
        return self._alpha_bar_clipped(t)

    def ddim_step(
        self,
        x_t: torch.Tensor,
        eps_pred: torch.Tensor,
        t_now: float,
        t_next: float,
    ) -> torch.Tensor:
        """Deterministic DDIM reverse step (eta=0).

        Args:
            x_t: Current noisy data.
            eps_pred: Predicted noise (epsilon).
            t_now: Current timestep.
            t_next: Next timestep (smaller). If <= 0, returns Tweedie estimate.

        Returns:
            Denoised data at t_next.
        """
        ab_now = max(self.alpha_bar(t_now), 1e-8)
        x_0_hat = (x_t - np.sqrt(1 - ab_now) * eps_pred) / np.sqrt(ab_now)
        if t_next <= 0:
            return x_0_hat
        ab_next = max(self.alpha_bar(t_next), 1e-8)
        return np.sqrt(ab_next) * x_0_hat + np.sqrt(1 - ab_next) * eps_pred

    def reverse_step(
        self,
        x_t: torch.Tensor,
        score: torch.Tensor,
        t: float,
        dt: float,
    ) -> torch.Tensor:
        """One reverse SDE step (Euler-Maruyama). DEPRECATED: use ddim_step."""
        beta_t = self.beta(t)
        drift = -0.5 * beta_t * x_t - beta_t * score
        diffusion = np.sqrt(max(beta_t, 1e-8))
        noise = torch.randn_like(x_t)
        x_prev = x_t + drift * dt + diffusion * np.sqrt(abs(dt)) * noise
        return x_prev
