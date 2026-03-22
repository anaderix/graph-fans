"""Core spectral profiling for Phase 0: Empirical Spectral Profiling."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


@dataclass
class SpectralProfile:
    """Results of spectral energy profiling on a graph."""

    name: str
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    band_boundaries: list[tuple[float, float]]
    band_energies: np.ndarray
    normalized_band_energies: np.ndarray
    band_indices: list[np.ndarray]
    energy_ratio: float
    decay_model: str  # 'power_law', 'exponential', or 'none'
    decay_params: dict
    decay_r_squared: float


def compute_laplacian_spectrum(
    graph: nx.Graph,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized Laplacian eigendecomposition.

    Returns:
        eigenvalues: sorted eigenvalues, shape [n]
        eigenvectors: corresponding eigenvectors, shape [n, n]
    """
    L = nx.normalized_laplacian_matrix(graph).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    # Clamp small negative eigenvalues from numerical noise
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return eigenvalues, eigenvectors


def partition_into_bands(
    eigenvalues: np.ndarray, B: int = 8, method: str = "uniform"
) -> tuple[list[tuple[float, float]], list[np.ndarray]]:
    """Partition eigenvalues into B bands.

    Args:
        eigenvalues: Sorted eigenvalues.
        B: Number of bands.
        method: 'uniform' (equal-width) or 'quantile' (equal-count).

    Returns:
        band_boundaries: List of (low, high) for each band.
        band_indices: List of index arrays for eigenvalues in each band.
    """
    n = len(eigenvalues)
    lambda_max = eigenvalues[-1] if eigenvalues[-1] > 0 else 1.0

    if method == "uniform":
        edges = np.linspace(0, lambda_max, B + 1)
    elif method == "quantile":
        edges = np.quantile(eigenvalues, np.linspace(0, 1, B + 1))
        # Ensure edges are unique and monotonically increasing
        edges = np.unique(edges)
        if len(edges) < B + 1:
            # Fall back to uniform if not enough unique quantiles
            edges = np.linspace(0, lambda_max, B + 1)
    else:
        raise ValueError(f"Unknown method: {method}")

    band_boundaries = []
    band_indices = []

    for b in range(len(edges) - 1):
        low, high = edges[b], edges[b + 1]
        if b == len(edges) - 2:
            # Last band includes the upper boundary
            mask = (eigenvalues >= low) & (eigenvalues <= high)
        else:
            mask = (eigenvalues >= low) & (eigenvalues < high)
        band_boundaries.append((low, high))
        band_indices.append(np.where(mask)[0])

    return band_boundaries, band_indices


def compute_band_energy(
    features: np.ndarray,
    eigenvectors: np.ndarray,
    band_indices: list[np.ndarray],
) -> np.ndarray:
    """Compute signal energy per spectral band.

    Projects node features onto eigenvectors in each band and computes
    energy (sum of squared coefficients) per band.

    Args:
        features: Node features, shape [n_nodes, n_features].
        eigenvectors: Eigenvector matrix, shape [n_nodes, n_nodes].
        band_indices: List of index arrays for eigenvalues in each band.

    Returns:
        band_energies: Energy per band, shape [B].
    """
    # Project features onto eigenbasis: coefficients = U^T @ features
    coefficients = eigenvectors.T @ features  # shape [n_nodes, n_features]

    band_energies = np.zeros(len(band_indices))
    for b, indices in enumerate(band_indices):
        if len(indices) > 0:
            band_energies[b] = np.sum(coefficients[indices] ** 2)

    return band_energies


def compute_energy_ratio(band_energies: np.ndarray) -> float:
    """Compute ratio of max to min non-zero band energy.

    This is the key G0 metric: ratio ≥ 2× indicates non-uniform spectral energy.
    """
    nonzero = band_energies[band_energies > 1e-10]
    if len(nonzero) < 2:
        return 1.0
    return float(nonzero.max() / nonzero.min())


def _power_law(x: np.ndarray, a: float, alpha: float) -> np.ndarray:
    return a * (x + 1.0) ** (-alpha)


def _exponential(x: np.ndarray, a: float, beta: float) -> np.ndarray:
    return a * np.exp(-beta * x)


def fit_decay_model(
    band_energies: np.ndarray,
) -> tuple[str, dict, float]:
    """Fit power-law and exponential decay models to band energies.

    Returns:
        model_name: 'power_law', 'exponential', or 'none'
        params: Fit parameters
        r_squared: R² of the better fit
    """
    x = np.arange(len(band_energies), dtype=float)
    y = band_energies.copy()

    # Skip if energies are too uniform or have zeros
    if y.max() < 1e-10 or (y.max() / max(y.min(), 1e-10)) < 1.5:
        return "none", {}, 0.0

    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot < 1e-10:
        return "none", {}, 0.0

    results = {}

    # Try power-law fit
    try:
        popt, _ = curve_fit(_power_law, x, y, p0=[y[0], 1.0], maxfev=5000)
        y_pred = _power_law(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        results["power_law"] = ({"a": popt[0], "alpha": popt[1]}, r2)
    except (RuntimeError, ValueError):
        pass

    # Try exponential fit
    try:
        popt, _ = curve_fit(_exponential, x, y, p0=[y[0], 0.5], maxfev=5000)
        y_pred = _exponential(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        results["exponential"] = ({"a": popt[0], "beta": popt[1]}, r2)
    except (RuntimeError, ValueError):
        pass

    if not results:
        return "none", {}, 0.0

    best = max(results.items(), key=lambda kv: kv[1][1])
    return best[0], best[1][0], best[1][1]


def spectral_energy_profile(
    graph: nx.Graph,
    features: np.ndarray,
    B: int = 8,
    name: str = "",
    method: str = "uniform",
) -> SpectralProfile:
    """Full spectral energy profiling pipeline.

    Args:
        graph: Input graph.
        features: Node features, shape [n_nodes, n_features].
        B: Number of spectral bands.
        name: Name for this profile.
        method: Band partitioning method ('uniform' or 'quantile').

    Returns:
        SpectralProfile with all computed quantities.
    """
    logger.info(f"Computing spectral profile for {name} (n={graph.number_of_nodes()})")

    eigenvalues, eigenvectors = compute_laplacian_spectrum(graph)
    band_boundaries, band_indices = partition_into_bands(eigenvalues, B, method)
    band_energies = compute_band_energy(features, eigenvectors, band_indices)

    # Normalize
    total = band_energies.sum()
    normalized = band_energies / total if total > 0 else band_energies

    energy_ratio = compute_energy_ratio(band_energies)
    decay_model, decay_params, decay_r2 = fit_decay_model(band_energies)

    logger.info(
        f"  Energy ratio: {energy_ratio:.2f}, "
        f"Decay model: {decay_model} (R²={decay_r2:.3f})"
    )

    return SpectralProfile(
        name=name,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        band_boundaries=band_boundaries,
        band_energies=band_energies,
        normalized_band_energies=normalized,
        band_indices=band_indices,
        energy_ratio=energy_ratio,
        decay_model=decay_model,
        decay_params=decay_params,
        decay_r_squared=decay_r2,
    )
