"""Gaussian augmentation for creating training distributions from single feature matrices.

Given a single real feature matrix, generates N augmented samples by adding
scaled Gaussian noise. This is needed because the diffusion Trainer expects
a dataset of [N, n_nodes, n_features] samples but real graphs have only one
feature matrix.
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
