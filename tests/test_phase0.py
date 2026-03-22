"""Tests for Phase 0: Spectral Profiling."""

import networkx as nx
import numpy as np
import pytest

from graph_fans.utils.graph_generators import generate_sbm, generate_ba
from graph_fans.phase0.spectral_profiler import (
    compute_laplacian_spectrum,
    partition_into_bands,
    compute_band_energy,
    compute_energy_ratio,
    fit_decay_model,
    spectral_energy_profile,
)


class TestGraphGenerators:
    def test_sbm_node_count(self):
        gd = generate_sbm(n_nodes=100, n_communities=4, p_intra=0.3, p_inter=0.05)
        assert gd.graph.number_of_nodes() == 100
        assert gd.features.shape[0] == 100

    def test_sbm_has_edges(self):
        gd = generate_sbm(n_nodes=50, p_intra=0.5, p_inter=0.1)
        assert gd.graph.number_of_edges() > 0

    def test_ba_node_count(self):
        gd = generate_ba(n_nodes=100, m=3)
        assert gd.graph.number_of_nodes() == 100
        assert gd.features.shape[0] == 100

    def test_ba_has_edges(self):
        gd = generate_ba(n_nodes=50, m=5)
        assert gd.graph.number_of_edges() > 0

    def test_features_shape(self):
        gd = generate_sbm(n_nodes=80, n_features=8)
        assert gd.features.shape == (80, 8)


class TestSpectralProfiler:
    @pytest.fixture
    def path_graph(self):
        """Path graph with known eigenvalues: 2 - 2*cos(k*pi/n)."""
        n = 20
        g = nx.path_graph(n)
        features = np.eye(n)[:, :4]  # Simple features
        return g, features, n

    def test_laplacian_spectrum_sorted(self, path_graph):
        g, _, n = path_graph
        eigenvalues, eigenvectors = compute_laplacian_spectrum(g)
        assert len(eigenvalues) == n
        assert np.all(np.diff(eigenvalues) >= -1e-10)  # Sorted

    def test_laplacian_spectrum_nonneg(self, path_graph):
        g, _, _ = path_graph
        eigenvalues, _ = compute_laplacian_spectrum(g)
        assert np.all(eigenvalues >= 0)

    def test_path_graph_eigenvalues(self, path_graph):
        """Path graph has known eigenvalues."""
        g, _, n = path_graph
        eigenvalues, _ = compute_laplacian_spectrum(g)
        # Normalized Laplacian eigenvalues for path graph
        # First eigenvalue should be 0
        assert eigenvalues[0] < 1e-8

    def test_band_partitioning_uniform(self):
        eigenvalues = np.linspace(0, 2, 100)
        boundaries, indices = partition_into_bands(eigenvalues, B=4, method="uniform")
        assert len(boundaries) == 4
        assert len(indices) == 4
        # All eigenvalues should be assigned
        all_assigned = set()
        for idx in indices:
            all_assigned.update(idx.tolist())
        assert len(all_assigned) == 100

    def test_band_partitioning_quantile(self):
        eigenvalues = np.linspace(0, 2, 100)
        boundaries, indices = partition_into_bands(eigenvalues, B=4, method="quantile")
        assert len(boundaries) == 4
        # Quantile bands should have roughly equal counts
        counts = [len(idx) for idx in indices]
        assert all(c > 0 for c in counts)

    def test_energy_preservation(self, path_graph):
        """Total energy across bands should equal total energy of features."""
        g, features, n = path_graph
        eigenvalues, eigenvectors = compute_laplacian_spectrum(g)
        _, band_indices = partition_into_bands(eigenvalues, B=4)
        band_energies = compute_band_energy(features, eigenvectors, band_indices)

        # Total band energy = ||U^T @ features||^2 = ||features||^2 (U is orthogonal)
        total_band_energy = band_energies.sum()
        total_feature_energy = np.sum(features ** 2)
        np.testing.assert_allclose(total_band_energy, total_feature_energy, rtol=1e-6)

    def test_energy_ratio(self):
        energies = np.array([10.0, 5.0, 2.0, 1.0])
        ratio = compute_energy_ratio(energies)
        assert ratio == 10.0

    def test_energy_ratio_uniform(self):
        energies = np.array([1.0, 1.0, 1.0, 1.0])
        ratio = compute_energy_ratio(energies)
        assert ratio == 1.0

    def test_fit_decay_power_law(self):
        x = np.arange(8, dtype=float)
        y = 10.0 * (x + 1.0) ** (-1.5) + np.random.RandomState(42).randn(8) * 0.01
        model, params, r2 = fit_decay_model(y)
        assert model in ("power_law", "exponential")
        assert r2 > 0.8

    def test_spectral_energy_profile_full(self):
        gd = generate_sbm(n_nodes=50, p_intra=0.5, p_inter=0.02)
        profile = spectral_energy_profile(gd.graph, gd.features, B=4, name="test")
        assert profile.name == "test"
        assert len(profile.band_energies) == 4
        assert profile.energy_ratio >= 1.0
        assert profile.decay_model in ("power_law", "exponential", "none")
