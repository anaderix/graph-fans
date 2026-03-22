"""Core FANS mechanism adapted for graphs: importance weighting and noise shaping."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class ImportanceWeights:
    """Per-band importance weights derived from training set spectral profile."""

    band_energies: np.ndarray  # [B] raw mean band energies from Phase 0
    weights: np.ndarray  # [B] importance weights g_b
    alpha: float  # power exponent used
    epsilon: float  # stability constant


def compute_importance_weights(
    band_energies: np.ndarray,
    alpha: float = 1.0,
    epsilon: float = 1e-3,
) -> ImportanceWeights:
    """Compute FANS-style importance weights from empirical band energies.

    g_b = (pi_bar_b + epsilon)^(-alpha), where pi_bar_b is normalized band energy.
    High-energy bands get lower weights; low-energy (underrepresented) bands get boosted.

    Args:
        band_energies: Raw per-band energies from Phase 0 profiling.
        alpha: Power exponent. Higher = more aggressive boosting.
        epsilon: Stability constant to prevent division by zero.
    """
    total = band_energies.sum()
    if total < 1e-10:
        return ImportanceWeights(
            band_energies=band_energies,
            weights=np.ones(len(band_energies)),
            alpha=alpha,
            epsilon=epsilon,
        )

    pi_bar = band_energies / total  # normalized band energy
    weights = (pi_bar + epsilon) ** (-alpha)
    # Normalize weights so mean = 1 (preserves overall noise scale)
    weights = weights / weights.mean()

    return ImportanceWeights(
        band_energies=band_energies,
        weights=weights,
        alpha=alpha,
        epsilon=epsilon,
    )


def shape_noise(
    noise: torch.Tensor,
    eigenvectors: torch.Tensor,
    band_indices: list[np.ndarray],
    weights: ImportanceWeights,
) -> torch.Tensor:
    """Shape noise in the Laplacian eigenbasis with per-band importance weighting.

    1. Project noise into eigenbasis: coeffs = U^T @ noise
    2. Scale each band by sqrt(g_b)
    3. Normalize each band to unit variance (critical for stability)
    4. Project back: shaped = U @ coeffs_scaled

    Args:
        noise: Standard Gaussian noise [n_nodes, n_features].
        eigenvectors: Eigenvector matrix [n_nodes, n_nodes] (columns are eigenvectors).
        band_indices: List of index arrays for eigenvalues in each band.
        weights: Importance weights from compute_importance_weights.

    Returns:
        Shaped noise [n_nodes, n_features].
    """
    U = eigenvectors  # [n, n]
    coeffs = U.T @ noise  # [n, n_features]

    coeffs_scaled = coeffs.clone()
    for b, indices in enumerate(band_indices):
        if len(indices) == 0:
            continue
        idx = torch.tensor(indices, device=noise.device, dtype=torch.long)
        band_coeffs = coeffs[idx]  # [band_size, n_features]

        # Scale by sqrt(weight) so variance scales by weight
        scale = float(np.sqrt(weights.weights[b]))
        band_coeffs = band_coeffs * scale

        # Per-band unit-variance normalization (critical for training stability)
        std = band_coeffs.std()
        if std > 1e-8:
            band_coeffs = band_coeffs / std

        coeffs_scaled[idx] = band_coeffs

    shaped = U @ coeffs_scaled  # [n, n_features]
    return shaped


def _smooth_step(x: float) -> float:
    """Smooth step function (cosine schedule) for temporal ramp."""
    x = max(0.0, min(1.0, x))
    return 0.5 * (1.0 - np.cos(np.pi * x))


def shape_noise_with_temporal_ramp(
    noise: torch.Tensor,
    eigenvectors: torch.Tensor,
    band_indices: list[np.ndarray],
    weights: ImportanceWeights,
    t: float,
    t_knee: float,
) -> torch.Tensor:
    """Apply temporal ramp: uniform noise below t_knee, shaped noise above.

    phi(t) smoothly interpolates from 0 (uniform) to 1 (fully shaped).
    shaped_noise = phi(t) * spectrally_shaped + (1 - phi(t)) * uniform

    Args:
        noise: Standard Gaussian noise [n_nodes, n_features].
        eigenvectors: Eigenvector matrix [n_nodes, n_nodes].
        band_indices: List of index arrays per band.
        weights: Importance weights.
        t: Current diffusion timestep in [0, 1].
        t_knee: Transition timestep.

    Returns:
        Temporally-ramped shaped noise [n_nodes, n_features].
    """
    if t <= t_knee:
        return noise

    phi = _smooth_step((t - t_knee) / (1.0 - t_knee))
    shaped = shape_noise(noise, eigenvectors, band_indices, weights)
    return phi * shaped + (1.0 - phi) * noise
