"""Spectral fidelity loss: directly penalize per-band energy mismatch.

Instead of shaping noise and hoping the model learns spectral structure,
we add an auxiliary loss that explicitly supervises per-band energy preservation.

The one-step denoised estimate (Tweedie's formula):
    x_hat_0 = (x_t + std^2 * score) / mean_coeff

Then: L_spectral = sum_b w_b * |E_b(x_hat_0) - E_b(x_0)| / E_total(x_0)
"""

from __future__ import annotations

import torch
import numpy as np

from .noise_shaper import ImportanceWeights


def compute_band_energy_torch(
    features: torch.Tensor,
    eigenvectors: torch.Tensor,
    band_indices: list[np.ndarray],
) -> torch.Tensor:
    """Compute per-band signal energy (differentiable).

    Args:
        features: [n_nodes, n_features]
        eigenvectors: [n_nodes, n_nodes]
        band_indices: list of index arrays per band

    Returns:
        band_energies: [B] tensor of per-band energies
    """
    coeffs = eigenvectors.T @ features  # [n, n_features]
    B = len(band_indices)
    energies = torch.zeros(B, device=features.device)

    for b, indices in enumerate(band_indices):
        if len(indices) == 0:
            continue
        idx = torch.tensor(indices, device=features.device, dtype=torch.long)
        energies[b] = (coeffs[idx] ** 2).sum()

    return energies


def spectral_fidelity_loss(
    x_hat_0: torch.Tensor,
    x_0: torch.Tensor,
    eigenvectors: torch.Tensor,
    band_indices: list[np.ndarray],
    weights: ImportanceWeights | None = None,
) -> torch.Tensor:
    """Compute spectral fidelity loss between denoised estimate and clean data.

    L = sum_b w_b * |E_b(x_hat_0) - E_b(x_0)| / E_total(x_0)

    Args:
        x_hat_0: One-step denoised estimate [n_nodes, n_features].
        x_0: Clean data [n_nodes, n_features].
        eigenvectors: [n_nodes, n_nodes].
        band_indices: List of index arrays per band.
        weights: Optional importance weights. If None, uniform weighting.

    Returns:
        Scalar loss value.
    """
    energy_pred = compute_band_energy_torch(x_hat_0, eigenvectors, band_indices)
    energy_ref = compute_band_energy_torch(x_0, eigenvectors, band_indices)

    # Normalize by total reference energy
    total_ref = energy_ref.sum().clamp(min=1e-8)
    energy_pred_norm = energy_pred / total_ref
    energy_ref_norm = energy_ref / total_ref

    # Weighted absolute difference
    B = len(band_indices)
    if weights is not None:
        w = torch.tensor(weights.weights, device=x_0.device, dtype=torch.float32)
    else:
        w = torch.ones(B, device=x_0.device)

    loss = (w * (energy_pred_norm - energy_ref_norm).abs()).sum()
    return loss


def tweedie_denoise(
    x_t: torch.Tensor,
    score: torch.Tensor,
    mean_coeff: float,
    std: float,
) -> torch.Tensor:
    """One-step denoised estimate via Tweedie's formula.

    x_hat_0 = (x_t + std^2 * score) / mean_coeff

    Args:
        x_t: Noisy data [n_nodes, n_features].
        score: Predicted score [n_nodes, n_features].
        mean_coeff: Marginal mean coefficient at time t.
        std: Marginal std at time t.

    Returns:
        Denoised estimate [n_nodes, n_features].
    """
    if mean_coeff < 1e-8:
        return x_t
    return (x_t + std**2 * score) / mean_coeff
