"""Spectral-domain Wasserstein distance for comparing feature distributions.

Projects features onto the Laplacian eigenbasis, then computes per-eigenmode
1D Wasserstein-1 distance between generated and reference sample distributions.
This captures distributional differences that mean-profile metrics (QBE) miss.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance


def spectral_coefficients(
    features: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """Project features onto Laplacian eigenbasis.

    Args:
        features: [n_nodes, n_features]
        eigenvectors: [n_nodes, n_nodes]

    Returns:
        coefficients: [n_nodes, n_features] — spectral coefficients.
    """
    return eigenvectors.T @ features


def per_mode_w1(
    features_ref: np.ndarray,
    features_gen: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """Compute per-eigenmode W1 distance between two sample sets.

    For each eigenmode k, we collect its energy (sum of squared coefficients
    across features) for every sample, giving two 1D distributions.
    Then compute W1 between them.

    Args:
        features_ref: [N_ref, n_nodes, n_features]
        features_gen: [N_gen, n_nodes, n_features]
        eigenvectors: [n_nodes, n_nodes]

    Returns:
        w1_per_mode: [n_nodes] — W1 distance for each eigenmode.
    """
    n_modes = eigenvectors.shape[1]

    # Collect per-mode energy for each sample
    def mode_energies(dataset: np.ndarray) -> np.ndarray:
        """Returns [N_samples, n_modes] — energy per mode per sample."""
        energies = []
        for features in dataset:
            coeffs = spectral_coefficients(features, eigenvectors)  # [n_modes, n_features]
            energies.append(np.sum(coeffs ** 2, axis=1))  # [n_modes]
        return np.stack(energies)  # [N, n_modes]

    ref_energies = mode_energies(features_ref)
    gen_energies = mode_energies(features_gen)

    w1 = np.zeros(n_modes)
    for k in range(n_modes):
        w1[k] = wasserstein_distance(ref_energies[:, k], gen_energies[:, k])

    return w1


def per_band_w1(
    features_ref: np.ndarray,
    features_gen: np.ndarray,
    eigenvectors: np.ndarray,
    band_indices: list[np.ndarray],
) -> np.ndarray:
    """Compute per-band W1 distance (aggregated over eigenmodes in each band).

    For each band, sums energy across its eigenmodes per sample, then computes
    W1 between the two resulting 1D distributions.

    Args:
        features_ref: [N_ref, n_nodes, n_features]
        features_gen: [N_gen, n_nodes, n_features]
        eigenvectors: [n_nodes, n_nodes]
        band_indices: List of index arrays per band.

    Returns:
        w1_per_band: [B] — W1 distance for each band.
    """

    def band_energies(dataset: np.ndarray) -> np.ndarray:
        """Returns [N_samples, B]."""
        energies = []
        for features in dataset:
            coeffs = spectral_coefficients(features, eigenvectors)  # [n_modes, n_features]
            mode_energy = np.sum(coeffs ** 2, axis=1)  # [n_modes]
            band_e = np.array([mode_energy[idx].sum() if len(idx) > 0 else 0.0
                               for idx in band_indices])
            energies.append(band_e)
        return np.stack(energies)  # [N, B]

    ref_energies = band_energies(features_ref)
    gen_energies = band_energies(features_gen)

    B = len(band_indices)
    w1 = np.zeros(B)
    for b in range(B):
        w1[b] = wasserstein_distance(ref_energies[:, b], gen_energies[:, b])

    return w1


def spectral_w1_summary(
    features_ref: np.ndarray,
    features_gen: np.ndarray,
    eigenvectors: np.ndarray,
    band_indices: list[np.ndarray],
) -> dict:
    """Full spectral W1 diagnostic.

    Returns:
        Dict with 'per_band_w1', 'total_w1', 'per_mode_w1',
        'low_band_w1' (mean of first half), 'high_band_w1' (mean of second half).
    """
    bw1 = per_band_w1(features_ref, features_gen, eigenvectors, band_indices)
    mw1 = per_mode_w1(features_ref, features_gen, eigenvectors)

    B = len(band_indices)
    half = B // 2

    return {
        "per_band_w1": bw1,
        "total_w1": float(bw1.sum()),
        "per_mode_w1": mw1,
        "low_band_w1": float(bw1[:half].mean()),
        "high_band_w1": float(bw1[half:].mean()),
    }
