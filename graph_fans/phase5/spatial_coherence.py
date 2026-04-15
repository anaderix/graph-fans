"""Spatial coherence metrics for evaluating spectral-domain generation quality.

Independent per-mode diffusion may produce features with correct spectral energy
but lacking spatial structure (neighbor correlations). These metrics detect that.
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def node_neighbor_correlation(features: np.ndarray, graph: nx.Graph) -> float:
    """Average cosine similarity of feature vectors between adjacent nodes.

    High values indicate spatially smooth features (typical of community structure).
    Random noise gives ~0.0; community features typically give 0.3-0.8.

    Args:
        features: [n_nodes, n_features]
        graph: nx.Graph

    Returns:
        Mean cosine similarity across all edges.
    """
    edges = list(graph.edges())
    if len(edges) == 0:
        return 0.0

    sims = []
    for u, v in edges:
        fu, fv = features[u], features[v]
        norm_u = np.linalg.norm(fu)
        norm_v = np.linalg.norm(fv)
        if norm_u > 1e-10 and norm_v > 1e-10:
            cos_sim = float(np.dot(fu, fv) / (norm_u * norm_v))
            sims.append(cos_sim)

    return float(np.mean(sims)) if sims else 0.0


def cross_mode_energy_correlation(
    features: np.ndarray,
    eigenvectors: np.ndarray,
) -> float:
    """Spearman correlation of per-mode energy profile vs expected ordering.

    For community features, energy should decrease with mode index (low-frequency
    modes carry most energy). A positive correlation with -rank indicates this.

    Args:
        features: [n_nodes, n_features]
        eigenvectors: [n_nodes, n_modes]

    Returns:
        Spearman rank correlation of energy profile vs negative mode index.
    """
    coeffs = eigenvectors.T @ features  # [n_modes, n_features]
    mode_energy = (coeffs ** 2).sum(axis=1)  # [n_modes]

    if len(mode_energy) < 3:
        return 0.0

    ranks = np.arange(len(mode_energy))
    corr, _ = spearmanr(mode_energy, -ranks)  # negative: high energy at low index

    if np.isnan(corr):
        return 0.0
    return float(corr)


def spatial_coherence_summary(
    ref_features: np.ndarray,
    gen_features: np.ndarray,
    graph: nx.Graph,
    eigenvectors: np.ndarray,
) -> dict:
    """Compute spatial coherence comparison between reference and generated features.

    Args:
        ref_features: [n_nodes, n_features] single reference sample.
        gen_features: [n_nodes, n_features] single generated sample.
        graph: nx.Graph.
        eigenvectors: [n_nodes, n_modes].

    Returns:
        Dict with neighbor correlation and mode energy correlation for both.
    """
    return {
        "ref_neighbor_corr": node_neighbor_correlation(ref_features, graph),
        "gen_neighbor_corr": node_neighbor_correlation(gen_features, graph),
        "ref_mode_energy_corr": cross_mode_energy_correlation(ref_features, eigenvectors),
        "gen_mode_energy_corr": cross_mode_energy_correlation(gen_features, eigenvectors),
    }
