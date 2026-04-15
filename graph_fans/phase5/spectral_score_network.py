"""Shared MLP for per-mode epsilon prediction in spectral diffusion.

Conditioned on eigenvalue lambda_k, diffusion time t, and mode energy E_k.
All modes are batched together for efficient forward pass -- no Python loop
over modes during training or generation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SpectralScoreNetwork(nn.Module):
    """Shared MLP for per-mode epsilon prediction.

    Architecture: 3-layer MLP with LayerNorm + SiLU activations.
    Input: concat [c_k_t(d), lambda_k(1), t(1), E_k(1)] = d+3 dims.
    Output: eps_pred [d] per mode.

    All modes are batched into a single forward pass for GPU efficiency.
    """

    def __init__(
        self,
        n_features: int = 4,
        hidden_dim: int = 128,
        n_layers: int = 3,
    ):
        super().__init__()
        input_dim = n_features + 3  # c_k_t (d) + lambda_k (1) + t (1) + E_k (1)
        layers: list[nn.Module] = []
        for i in range(n_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_d, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ])
        layers.append(nn.Linear(hidden_dim, n_features))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        c_k_t: Tensor,
        lambda_k: Tensor,
        t: Tensor,
        E_k: Tensor,
    ) -> Tensor:
        """Predict epsilon for all modes in batch.

        Args:
            c_k_t: [batch, d] noisy spectral coefficients.
            lambda_k: [batch, 1] eigenvalues.
            t: [batch, 1] diffusion times.
            E_k: [batch, 1] mode energies (from training data).

        Returns:
            eps_pred: [batch, d] predicted noise.
        """
        x = torch.cat([c_k_t, lambda_k, t, E_k], dim=-1)
        return self.net(x)
