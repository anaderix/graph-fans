"""Multi-scale, structure-aware feature generation for graph diffusion experiments.

The key insight: for spectral noise shaping to matter, features must have
scale-dependent spectral content. A hub node, a cluster-interior node, and
a bridge node should contribute energy to different spectral bands.

This module generates features as a mixture of spectral components where
each node's mixing weights depend on its structural role in the graph.
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
from scipy.linalg import expm

logger = logging.getLogger(__name__)


def compute_structural_roles(graph: nx.Graph) -> dict[str, np.ndarray]:
    """Compute per-node structural role indicators.

    Returns dict with arrays of shape [n_nodes]:
        - 'centrality': eigenvector centrality (high for hubs)
        - 'boundary': participation coefficient (high for bridge nodes)
        - 'locality': inverse of centrality (high for peripheral nodes)
    """
    n = graph.number_of_nodes()

    # Eigenvector centrality (hubs score high)
    try:
        ec = nx.eigenvector_centrality_numpy(graph)
        centrality = np.array([ec[i] for i in range(n)])
    except Exception:
        centrality = np.ones(n) / n

    # Normalize to [0, 1]
    cmax = centrality.max()
    if cmax > 1e-10:
        centrality = centrality / cmax

    # Boundary score: based on neighbor diversity
    # High when a node's neighbors span multiple communities/clusters
    # We approximate this using the variation in neighbor degrees
    boundary = np.zeros(n)
    for node in range(n):
        neighbors = list(graph.neighbors(node))
        if len(neighbors) < 2:
            boundary[node] = 0.0
            continue
        # Variation in neighbor centrality = structural boundary indicator
        neighbor_c = centrality[neighbors]
        boundary[node] = neighbor_c.std() / (neighbor_c.mean() + 1e-10)

    bmax = boundary.max()
    if bmax > 1e-10:
        boundary = boundary / bmax

    # Locality: complement of centrality (high for peripheral/leaf nodes)
    locality = 1.0 - centrality

    return {
        "centrality": centrality,
        "boundary": boundary,
        "locality": locality,
    }


def generate_multiscale_features(
    graph: nx.Graph,
    n_features: int = 16,
    seed: int = 42,
    n_scales: int = 3,
) -> np.ndarray:
    """Generate multi-scale, structure-aware node features.

    Creates features as a mixture of spectral components at different scales.
    Each node's mixing weights depend on its structural role:

    - **Low-frequency component** (community signal): weighted by locality
      Interior cluster nodes get strong smooth community features.

    - **Mid-frequency component** (boundary signal): weighted by boundary score
      Bridge nodes between communities get transitional features.

    - **High-frequency component** (local detail): weighted by centrality
      Hubs get high-frequency content reflecting diverse neighborhoods.

    The result is a feature matrix where the spectral energy profile
    genuinely varies across nodes, creating the multi-modal distribution
    that spectral noise shaping should help preserve.

    Args:
        graph: Input graph.
        n_features: Number of features per node.
        seed: Random seed.
        n_scales: Number of spectral scales (default 3: low, mid, high).

    Returns:
        features: [n_nodes, n_features] array.
    """
    rng = np.random.RandomState(seed)
    n = graph.number_of_nodes()
    L = nx.normalized_laplacian_matrix(graph).toarray()

    # Eigendecomposition
    eigenvalues, U = np.linalg.eigh(L)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    lambda_max = eigenvalues[-1] if eigenvalues[-1] > 0 else 1.0

    # Structural roles
    roles = compute_structural_roles(graph)

    # Define scale filters via heat kernels at different temperatures
    # Low t = high-pass (local detail), High t = low-pass (global/community)
    t_values = np.logspace(np.log10(0.1), np.log10(10.0), n_scales)  # [0.1, 1.0, 10.0]

    # Generate random spectral coefficients for each scale
    features = np.zeros((n, n_features))

    for scale_idx, t in enumerate(t_values):
        # Heat kernel filter at this scale
        # exp(-t * lambda_i): large t → low-pass, small t → keeps more high-freq
        filter_response = np.exp(-t * eigenvalues)

        # Generate random coefficients in eigenbasis
        raw_coeffs = rng.randn(n, n_features)

        # Apply spectral filter
        filtered_coeffs = raw_coeffs * filter_response[:, np.newaxis]

        # Project back to node space
        scale_features = U @ filtered_coeffs

        # Compute per-node mixing weight based on structural role
        if scale_idx == 0:
            # High-frequency (small t): hubs and central nodes
            weight = roles["centrality"]
        elif scale_idx == n_scales - 1:
            # Low-frequency (large t): interior/peripheral nodes
            weight = roles["locality"]
        else:
            # Mid-frequency: boundary/bridge nodes
            weight = roles["boundary"]

        # Ensure minimum weight so no scale is completely absent
        weight = 0.2 + 0.8 * weight  # range [0.2, 1.0]

        # Apply per-node weighting
        features += scale_features * weight[:, np.newaxis]

    # Normalize per feature to unit variance
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-10] = 1.0
    features = features / std

    return features


def generate_community_boundary_features(
    graph: nx.Graph,
    n_features: int = 16,
    seed: int = 42,
    community_labels: np.ndarray | None = None,
) -> np.ndarray:
    """Generate features with explicit community and boundary structure.

    An alternative to the heat-kernel approach: directly assigns features
    based on community membership with sharp transitions at boundaries.

    - Within-community nodes: features = community_centroid + small noise
    - Boundary nodes: features = interpolation between neighboring community centroids
    - Hub nodes: features = mixture of all community centroids + unique high-freq signature

    This creates features where:
    - Low eigenvalue bands encode community identity (smooth within communities)
    - Mid eigenvalue bands encode boundary transitions
    - High eigenvalue bands encode hub diversity and node-level variation

    Args:
        graph: Input graph.
        n_features: Number of features per node.
        seed: Random seed.
        community_labels: Optional pre-computed community labels [n_nodes].
            If None, detected via Louvain or spectral clustering.
    """
    rng = np.random.RandomState(seed)
    n = graph.number_of_nodes()

    # Detect communities if not provided
    if community_labels is None:
        try:
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(graph, seed=seed)
            community_labels = np.zeros(n, dtype=int)
            for idx, comm in enumerate(communities):
                for node in comm:
                    community_labels[node] = idx
        except Exception:
            # Fallback: spectral clustering via Fiedler vector
            L = nx.normalized_laplacian_matrix(graph).toarray()
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            fiedler = eigenvectors[:, 1]
            community_labels = (fiedler > np.median(fiedler)).astype(int)

    n_communities = len(np.unique(community_labels))

    # Generate distinct community centroids (well-separated in feature space)
    centroids = rng.randn(n_communities, n_features)
    # Orthogonalize for maximum separation
    if n_communities <= n_features:
        from scipy.linalg import qr
        Q, _ = qr(centroids.T, mode="economic")
        centroids = Q.T[:n_communities] * 3.0  # scale for separation

    # Compute per-node roles
    roles = compute_structural_roles(graph)

    features = np.zeros((n, n_features))
    for node in range(n):
        comm = community_labels[node]

        # Base: community centroid
        base = centroids[comm].copy()

        # Boundary contribution: interpolate with neighboring communities
        neighbors = list(graph.neighbors(node))
        if neighbors:
            neighbor_comms = community_labels[neighbors]
            unique_comms = np.unique(neighbor_comms)
            if len(unique_comms) > 1:
                # Node is at a boundary — mix in other community centroids
                boundary_mix = np.zeros(n_features)
                for nc in unique_comms:
                    if nc != comm:
                        frac = (neighbor_comms == nc).mean()
                        boundary_mix += frac * centroids[nc]
                # Scale by boundary score
                base += roles["boundary"][node] * boundary_mix

        # Hub contribution: add high-frequency signature proportional to centrality
        hub_noise = rng.randn(n_features) * roles["centrality"][node] * 0.5

        # Node-level noise (small, unique per node)
        node_noise = rng.randn(n_features) * 0.1

        features[node] = base + hub_noise + node_noise

    # Normalize per feature to unit variance
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-10] = 1.0
    features = features / std

    return features
