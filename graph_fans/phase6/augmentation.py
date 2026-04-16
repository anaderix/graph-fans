"""Augmentation strategies for creating training distributions from single feature matrices.

Given a single real feature matrix, generates N augmented samples. This is
needed because the diffusion Trainer expects a dataset of [N, n_nodes, n_features]
samples but real graphs have only one feature matrix.

Strategies:
- gaussian: Additive Gaussian noise scaled by feature std.
- spectral: Add noise in eigenbasis proportional to per-mode energy.
- dropout: Randomly zero features.
"""

from __future__ import annotations

import numpy as np


def gaussian_augmentation(
    features: np.ndarray,
    n_samples: int = 100,
    sigma_range: tuple[float, float] = (0.01, 0.5),
    seed: int = 0,
) -> np.ndarray:
    """Create N augmented samples by adding scaled Gaussian noise.

    For each sample i, draws sigma_i ~ U(sigma_range[0], sigma_range[1]) * std(features),
    then x_aug_i = features + sigma_i * randn(n_nodes, n_features).

    Args:
        features: Original feature matrix, shape [n_nodes, n_features].
        n_samples: Number of augmented samples to create.
        sigma_range: Range of noise scale as fraction of feature std.
            sigma_i = U(sigma_range[0], sigma_range[1]) * std(features).
        seed: Random seed for reproducibility.

    Returns:
        Augmented samples, shape [n_samples, n_nodes, n_features].
    """
    rng = np.random.RandomState(seed)
    feat_std = features.std()
    if feat_std < 1e-10:
        feat_std = 1.0

    n_nodes, n_features = features.shape
    augmented = np.empty((n_samples, n_nodes, n_features), dtype=np.float64)

    for i in range(n_samples):
        sigma_i = rng.uniform(sigma_range[0], sigma_range[1]) * feat_std
        noise = rng.randn(n_nodes, n_features) * sigma_i
        augmented[i] = features + noise

    return augmented


def spectral_augmentation(
    features: np.ndarray,
    eigenvectors: np.ndarray,
    n_samples: int = 100,
    sigma: float = 0.3,
    seed: int = 0,
) -> np.ndarray:
    """Create N augmented samples by adding noise in eigenbasis proportional to mode energy.

    Projects features to spectral coefficients, then adds noise scaled by
    the relative energy of each mode. This preserves the spectral profile
    of the original features better than isotropic Gaussian noise.

    Args:
        features: Original feature matrix, shape [n_nodes, n_features].
        eigenvectors: Laplacian eigenvectors, shape [n_nodes, n_nodes].
        n_samples: Number of augmented samples to create.
        sigma: Base noise scale.
        seed: Random seed for reproducibility.

    Returns:
        Augmented samples, shape [n_samples, n_nodes, n_features].
    """
    rng = np.random.RandomState(seed)
    coeffs = eigenvectors.T @ features  # [n_modes, n_features]
    mode_energies = (coeffs ** 2).sum(axis=1)  # [n_modes]
    scale = np.sqrt(mode_energies / mode_energies.max() + 1e-8)  # relative scale per mode

    samples = []
    for i in range(n_samples):
        noise_coeffs = coeffs + sigma * scale[:, None] * rng.randn(*coeffs.shape)
        samples.append(eigenvectors @ noise_coeffs)

    return np.stack(samples)


def dropout_augmentation(
    features: np.ndarray,
    n_samples: int = 100,
    drop_rate: float = 0.2,
    seed: int = 0,
) -> np.ndarray:
    """Create N augmented samples by randomly zeroing features.

    For each sample, independently drops each feature entry with probability
    drop_rate. No rescaling is applied (unlike inverted dropout) since the
    goal is to create diverse training distribution, not preserve expectations.

    Args:
        features: Original feature matrix, shape [n_nodes, n_features].
        n_samples: Number of augmented samples to create.
        drop_rate: Probability of zeroing each feature entry.
        seed: Random seed for reproducibility.

    Returns:
        Augmented samples, shape [n_samples, n_nodes, n_features].
    """
    rng = np.random.RandomState(seed)
    samples = []
    for i in range(n_samples):
        mask = rng.random(features.shape) > drop_rate
        samples.append(features * mask)

    return np.stack(samples)
