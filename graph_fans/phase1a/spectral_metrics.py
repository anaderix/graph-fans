"""Basis-free spectral evaluation metrics for Phase 1a.

Implements three metrics:
1. Spectral Density Distance (JSD on eigenvalue KDEs)
2. Quantile-Band Energy Distance
3. Heat Kernel Signature Distance

Plus standard MMD metrics for comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde

logger = logging.getLogger(__name__)


# =============================================================================
# Metric 1: Spectral Density Distance (JSD)
# =============================================================================


def spectral_density_jsd(
    eigenvalues_ref: np.ndarray,
    eigenvalues_gen: np.ndarray,
    n_points: int = 1000,
) -> float:
    """Compute Jensen-Shannon divergence between spectral density estimates.

    Normalizes eigenvalues by λ_max, estimates density via KDE, then
    computes JSD on a shared evaluation grid.
    """
    # Normalize by lambda_max
    lmax_ref = eigenvalues_ref.max()
    lmax_gen = eigenvalues_gen.max()
    if lmax_ref < 1e-10 or lmax_gen < 1e-10:
        return 1.0

    ev_ref = eigenvalues_ref / lmax_ref
    ev_gen = eigenvalues_gen / lmax_gen

    # Filter out zero eigenvalues (trivial)
    ev_ref = ev_ref[ev_ref > 1e-8]
    ev_gen = ev_gen[ev_gen > 1e-8]

    if len(ev_ref) < 2 or len(ev_gen) < 2:
        return 1.0

    # KDE estimation
    try:
        kde_ref = gaussian_kde(ev_ref, bw_method="silverman")
        kde_gen = gaussian_kde(ev_gen, bw_method="silverman")
    except np.linalg.LinAlgError:
        return 1.0

    # Evaluation grid
    grid = np.linspace(0, 1, n_points)
    p = kde_ref(grid)
    q = kde_gen(grid)

    # Normalize to probability distributions
    p = p / (p.sum() + 1e-10)
    q = q / (q.sum() + 1e-10)

    return float(jensenshannon(p, q) ** 2)  # JSD (squared, as is conventional)


# =============================================================================
# Metric 2: Quantile-Band Energy
# =============================================================================


def quantile_band_energy(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    features: np.ndarray,
    B: int = 8,
) -> np.ndarray:
    """Compute signal energy per quantile band.

    Defines B bands as equal-mass quantiles of eigenvalue distribution,
    so each band contains ~n/B eigenvalues.

    Returns:
        Per-band energy array, shape [B].
    """
    n = len(eigenvalues)
    quantile_edges = np.quantile(eigenvalues, np.linspace(0, 1, B + 1))

    coefficients = eigenvectors.T @ features  # [n, n_features]

    band_energies = np.zeros(B)
    for b in range(B):
        low, high = quantile_edges[b], quantile_edges[b + 1]
        if b == B - 1:
            mask = (eigenvalues >= low) & (eigenvalues <= high)
        else:
            mask = (eigenvalues >= low) & (eigenvalues < high)
        indices = np.where(mask)[0]
        if len(indices) > 0:
            band_energies[b] = np.sum(coefficients[indices] ** 2)

    return band_energies


def quantile_band_energy_distance(
    eigenvalues_ref: np.ndarray,
    eigenvectors_ref: np.ndarray,
    features_ref: np.ndarray,
    eigenvalues_gen: np.ndarray,
    eigenvectors_gen: np.ndarray,
    features_gen: np.ndarray,
    B: int = 8,
) -> float:
    """L2 distance between normalized quantile-band energy vectors."""
    e_ref = quantile_band_energy(eigenvalues_ref, eigenvectors_ref, features_ref, B)
    e_gen = quantile_band_energy(eigenvalues_gen, eigenvectors_gen, features_gen, B)

    # Normalize
    e_ref = e_ref / (e_ref.sum() + 1e-10)
    e_gen = e_gen / (e_gen.sum() + 1e-10)

    return float(np.linalg.norm(e_ref - e_gen))


# =============================================================================
# Metric 3: Heat Kernel Signature Distance
# =============================================================================


def heat_kernel_trace(
    eigenvalues: np.ndarray,
    t_values: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Tr(exp(-tL)) = sum_i exp(-t * λ_i) for each t.

    Returns array of trace values, shape [len(t_values)].
    """
    if t_values is None:
        t_values = np.logspace(-2, 2, 50)

    # [n_t, n_eigenvalues]
    traces = np.sum(
        np.exp(-np.outer(t_values, eigenvalues)), axis=1
    )
    return traces


def hks_distance(
    eigenvalues_ref: np.ndarray,
    eigenvalues_gen: np.ndarray,
    t_values: np.ndarray | None = None,
) -> float:
    """L2 distance between normalized HKS trace curves."""
    if t_values is None:
        t_values = np.logspace(-2, 2, 50)

    hks_ref = heat_kernel_trace(eigenvalues_ref, t_values)
    hks_gen = heat_kernel_trace(eigenvalues_gen, t_values)

    # Normalize by number of nodes
    n_ref = len(eigenvalues_ref)
    n_gen = len(eigenvalues_gen)
    hks_ref = hks_ref / n_ref
    hks_gen = hks_gen / n_gen

    return float(np.linalg.norm(hks_ref - hks_gen))


# =============================================================================
# Standard MMD metrics for comparison
# =============================================================================


def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """MMD with RBF kernel between two 1D distributions.

    Args:
        X, Y: 1D arrays of values (e.g., degree sequences).
        gamma: RBF kernel bandwidth parameter.
    """
    X = X.reshape(-1, 1).astype(float)
    Y = Y.reshape(-1, 1).astype(float)

    XX = np.exp(-gamma * (X - X.T) ** 2)
    YY = np.exp(-gamma * (Y - Y.T) ** 2)
    XY = np.exp(-gamma * (X - Y.T) ** 2)

    return float(XX.mean() + YY.mean() - 2 * XY.mean())


def degree_mmd(graph_ref: nx.Graph, graph_gen: nx.Graph, gamma: float = 1.0) -> float:
    """MMD on degree distributions."""
    d_ref = np.array([d for _, d in graph_ref.degree()])
    d_gen = np.array([d for _, d in graph_gen.degree()])
    return mmd_rbf(d_ref, d_gen, gamma)


def clustering_mmd(
    graph_ref: nx.Graph, graph_gen: nx.Graph, gamma: float = 1.0
) -> float:
    """MMD on clustering coefficient distributions."""
    c_ref = np.array(list(nx.clustering(graph_ref).values()))
    c_gen = np.array(list(nx.clustering(graph_gen).values()))
    if len(c_ref) == 0 or len(c_gen) == 0:
        return 1.0
    return mmd_rbf(c_ref, c_gen, gamma)


def triangle_count_mmd(
    graph_ref: nx.Graph, graph_gen: nx.Graph, gamma: float = 0.1
) -> float:
    """MMD on per-node triangle counts (proxy for orbit statistics)."""
    t_ref = np.array(list(nx.triangles(graph_ref).values()), dtype=float)
    t_gen = np.array(list(nx.triangles(graph_gen).values()), dtype=float)
    if len(t_ref) == 0 or len(t_gen) == 0:
        return 1.0
    return mmd_rbf(t_ref, t_gen, gamma)


# =============================================================================
# Wrapper class
# =============================================================================


@dataclass
class MetricResults:
    """Container for all metric results."""

    spectral_density_jsd: float
    quantile_band_energy_dist: float
    hks_distance: float
    degree_mmd: float
    clustering_mmd: float
    triangle_mmd: float

    def spectral_aggregate(self) -> float:
        """Aggregate spectral metric (mean of normalized values)."""
        return (self.spectral_density_jsd + self.quantile_band_energy_dist + self.hks_distance) / 3

    def standard_aggregate(self) -> float:
        """Aggregate standard MMD metric."""
        return (self.degree_mmd + self.clustering_mmd + self.triangle_mmd) / 3


class SpectralMetrics:
    """Compute all spectral and standard metrics between graphs."""

    def __init__(self, B: int = 8, t_values: np.ndarray | None = None):
        self.B = B
        self.t_values = t_values or np.logspace(-2, 2, 50)

    def compute_all(
        self,
        graph_ref: nx.Graph,
        features_ref: np.ndarray,
        graph_gen: nx.Graph,
        features_gen: np.ndarray,
    ) -> MetricResults:
        """Compute all spectral and standard metrics."""
        from graph_fans.phase0.spectral_profiler import compute_laplacian_spectrum

        ev_ref, U_ref = compute_laplacian_spectrum(graph_ref)
        ev_gen, U_gen = compute_laplacian_spectrum(graph_gen)

        jsd = spectral_density_jsd(ev_ref, ev_gen)
        qbe = quantile_band_energy_distance(
            ev_ref, U_ref, features_ref,
            ev_gen, U_gen, features_gen,
            B=self.B,
        )
        hks = hks_distance(ev_ref, ev_gen, self.t_values)

        d_mmd = degree_mmd(graph_ref, graph_gen)
        c_mmd = clustering_mmd(graph_ref, graph_gen)
        t_mmd = triangle_count_mmd(graph_ref, graph_gen)

        return MetricResults(
            spectral_density_jsd=jsd,
            quantile_band_energy_dist=qbe,
            hks_distance=hks,
            degree_mmd=d_mmd,
            clustering_mmd=c_mmd,
            triangle_mmd=t_mmd,
        )

    def compute_standard_metrics(
        self, graph_ref: nx.Graph, graph_gen: nx.Graph
    ) -> dict[str, float]:
        """Compute only standard MMD metrics."""
        return {
            "degree_mmd": degree_mmd(graph_ref, graph_gen),
            "clustering_mmd": clustering_mmd(graph_ref, graph_gen),
            "triangle_mmd": triangle_count_mmd(graph_ref, graph_gen),
        }
