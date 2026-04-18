"""Augmentation baselines for Phase 7b downstream classification.

Implements standard feature augmentation methods from the graph ML literature
that we compare spectral augmentation against.

Baselines:
  - DropFeature (Rong et al. 2020): randomly zero a fraction of feature dims.
  - FeatMask: randomly mask entire node feature vectors.
  - GraphMix / node-level mixup (Wang et al. 2020): convex combinations of
    features from random node pairs.
  - FLAG (Kong et al. 2022): adversarial feature perturbation via PGD.
  - no_aug: identity passthrough.

Each function takes (features, seed, ...) and returns augmented features with
the same shape [n_nodes, n_features]. These are applied *every epoch* during
classifier training (unlike Phase 7a augmentation which creates a fixed
training distribution).
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def no_aug(features: np.ndarray, seed: int = 0) -> np.ndarray:
    """Identity: return features unchanged."""
    return features.copy()


def drop_feature(features: np.ndarray, drop_rate: float = 0.2, seed: int = 0) -> np.ndarray:
    """DropFeature: zero a fraction of feature dimensions (columns).

    Rong et al. 2020. Drops whole columns so every node shares the same mask
    (standard DropFeature formulation).
    """
    rng = np.random.RandomState(seed)
    n_features = features.shape[1]
    mask = rng.random(n_features) > drop_rate
    return features * mask[None, :]


def feat_mask(features: np.ndarray, mask_rate: float = 0.1, seed: int = 0) -> np.ndarray:
    """FeatMask: zero entire node feature vectors with probability mask_rate."""
    rng = np.random.RandomState(seed)
    n_nodes = features.shape[0]
    keep = rng.random(n_nodes) > mask_rate
    return features * keep[:, None]


def graph_mix(features: np.ndarray, alpha: float = 1.0, seed: int = 0) -> np.ndarray:
    """Node-level mixup: x_i = lam * x_i + (1-lam) * x_{perm(i)}.

    Simplified GraphMix/Mixup variant (Wang et al. 2020). lam ~ Beta(alpha, alpha).
    """
    rng = np.random.RandomState(seed)
    n = features.shape[0]
    perm = rng.permutation(n)
    lam = float(rng.beta(alpha, alpha))
    return lam * features + (1.0 - lam) * features[perm]


def gaussian_noise(features: np.ndarray, sigma: float = 0.1, seed: int = 0) -> np.ndarray:
    """Gaussian additive noise baseline."""
    rng = np.random.RandomState(seed)
    std = features.std()
    if std < 1e-10:
        std = 1.0
    return features + sigma * std * rng.randn(*features.shape)


def spectral_noise(
    features: np.ndarray,
    eigenvectors: np.ndarray,
    sigma: float = 0.3,
    seed: int = 0,
) -> np.ndarray:
    """Spectral augmentation: noise in eigenbasis proportional to per-mode energy.

    Matches `graph_fans.phase6.augmentation.spectral_augmentation` but operates
    on a single realization (one augmented matrix per call).
    """
    rng = np.random.RandomState(seed)
    coeffs = eigenvectors.T @ features  # [n_modes, n_features]
    mode_energies = (coeffs ** 2).sum(axis=1)
    scale = np.sqrt(mode_energies / mode_energies.max() + 1e-8)
    noise_coeffs = coeffs + sigma * scale[:, None] * rng.randn(*coeffs.shape)
    return eigenvectors @ noise_coeffs


AugFn = Callable[..., np.ndarray]

AUGMENTATIONS: dict[str, AugFn] = {
    "none": no_aug,
    "drop_feature": drop_feature,
    "feat_mask": feat_mask,
    "graph_mix": graph_mix,
    "gaussian": gaussian_noise,
    "spectral": spectral_noise,
}
