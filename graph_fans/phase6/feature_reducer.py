"""PCA/TruncatedSVD wrapper for dimensionality reduction of citation features.

TruncatedSVD is preferred for sparse binary features (Cora) since it doesn't
center the data. PCA is available for continuous features (PubMed TF-IDF).
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD, PCA


def fit_feature_reducer(
    features: np.ndarray,
    n_components: int,
    method: str = "truncated_svd",
    seed: int = 0,
) -> TruncatedSVD | PCA:
    """Fit a dimensionality reducer on the full feature matrix.

    Args:
        features: Full feature matrix, shape [n_nodes, n_features].
        n_components: Number of output dimensions.
        method: "truncated_svd" (no centering, good for sparse binary) or
            "pca" (centers data, good for continuous features).
        seed: Random seed for reproducibility.

    Returns:
        Fitted sklearn transformer.
    """
    if method == "truncated_svd":
        reducer = TruncatedSVD(n_components=n_components, random_state=seed)
    elif method == "pca":
        reducer = PCA(n_components=n_components, random_state=seed)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'truncated_svd' or 'pca'.")

    reducer.fit(features)
    return reducer


def reduce_features(
    features: np.ndarray, reducer: TruncatedSVD | PCA
) -> np.ndarray:
    """Apply a fitted reducer to a feature subset.

    Args:
        features: Feature matrix, shape [n_nodes, n_features].
        reducer: Fitted sklearn transformer from fit_feature_reducer.

    Returns:
        Reduced features, shape [n_nodes, n_components].
    """
    return reducer.transform(features)
