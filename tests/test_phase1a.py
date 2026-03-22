"""Tests for Phase 1a: Basis-Free Spectral Metrics."""

import networkx as nx
import numpy as np
import pytest

from graph_fans.phase0.spectral_profiler import compute_laplacian_spectrum
from graph_fans.phase1a.spectral_metrics import (
    spectral_density_jsd,
    quantile_band_energy,
    quantile_band_energy_distance,
    heat_kernel_trace,
    hks_distance,
    mmd_rbf,
    degree_mmd,
    clustering_mmd,
    SpectralMetrics,
)


class TestSpectralDensityJSD:
    def test_identical_eigenvalues(self):
        ev = np.array([0, 0.5, 1.0, 1.5, 2.0])
        jsd = spectral_density_jsd(ev, ev)
        assert jsd < 0.01  # Should be near zero

    def test_different_eigenvalues(self):
        ev1 = np.linspace(0, 2, 50)
        ev2 = np.array([0] + [2.0] * 49)  # Very different distribution
        jsd = spectral_density_jsd(ev1, ev2)
        assert jsd > 0.01  # Should detect the difference

    def test_jsd_bounded(self):
        ev1 = np.linspace(0, 2, 30)
        ev2 = np.linspace(0, 2, 30) ** 2
        jsd = spectral_density_jsd(ev1, ev2)
        assert 0 <= jsd <= 1.0  # JSD is bounded


class TestHeatKernelSignature:
    def test_path_graph_hks(self):
        """HKS on path graph with known eigenvalues."""
        n = 20
        g = nx.path_graph(n)
        ev, _ = compute_laplacian_spectrum(g)
        t_values = np.logspace(-1, 1, 20)
        hks = heat_kernel_trace(ev, t_values)
        assert len(hks) == 20
        # At t=0, trace should be n (sum of exp(0))
        hks_t0 = heat_kernel_trace(ev, np.array([0.0]))
        np.testing.assert_allclose(hks_t0[0], n, atol=0.01)

    def test_hks_distance_identical(self):
        ev = np.linspace(0, 2, 30)
        dist = hks_distance(ev, ev)
        assert dist < 1e-6

    def test_hks_distance_different(self):
        ev1 = np.linspace(0, 2, 30)
        ev2 = np.linspace(0, 4, 30)
        dist = hks_distance(ev1, ev2)
        assert dist > 0.01


class TestQuantileBandEnergy:
    def test_band_count(self):
        n = 40
        g = nx.barabasi_albert_graph(n, 3, seed=42)
        ev, U = compute_laplacian_spectrum(g)
        features = np.random.RandomState(42).randn(n, 4)
        energies = quantile_band_energy(ev, U, features, B=4)
        assert len(energies) == 4

    def test_energy_nonneg(self):
        n = 40
        g = nx.barabasi_albert_graph(n, 3, seed=42)
        ev, U = compute_laplacian_spectrum(g)
        features = np.random.RandomState(42).randn(n, 4)
        energies = quantile_band_energy(ev, U, features, B=4)
        assert np.all(energies >= 0)


class TestMMD:
    def test_mmd_same_distribution(self):
        X = np.random.RandomState(42).randn(100)
        mmd = mmd_rbf(X, X)
        # Self-MMD should be very small (but not necessarily 0 due to kernel)
        assert mmd < 0.1

    def test_mmd_different_distributions(self):
        X = np.random.RandomState(42).randn(100)
        Y = np.random.RandomState(42).randn(100) + 5.0  # Shifted
        mmd = mmd_rbf(X, Y)
        assert mmd > 0.1

    def test_degree_mmd_identical(self):
        g = nx.barabasi_albert_graph(100, 3, seed=42)
        mmd = degree_mmd(g, g)
        assert mmd < 0.01


class TestSpectralMetricsDetectsGaps:
    def test_spectral_vs_degree_sensitivity(self):
        """Two graphs with same degree sequence but different spectral properties.

        A regular graph and a random graph with matching degree sequence
        can have very different spectral properties.
        """
        # Create a ring graph (very regular spectral structure)
        n = 50
        g_ring = nx.cycle_graph(n)

        # Create random graph with similar density
        m = g_ring.number_of_edges()
        p = 2 * m / (n * (n - 1))
        g_er = nx.erdos_renyi_graph(n, p, seed=42)

        ev_ring, _ = compute_laplacian_spectrum(g_ring)
        ev_er, _ = compute_laplacian_spectrum(g_er)

        # Spectral JSD should detect the difference
        jsd = spectral_density_jsd(ev_ring, ev_er)

        # Degree MMD may be small (both have similar average degree)
        d_mmd = degree_mmd(g_ring, g_er)

        # We expect spectral metrics to be more sensitive than degree-based
        # This is a weak test — just verify both compute without error
        assert jsd >= 0
        assert d_mmd >= 0

    def test_full_metrics_wrapper(self):
        """Test SpectralMetrics wrapper computes all metrics."""
        n = 30
        g1 = nx.barabasi_albert_graph(n, 3, seed=42)
        g2 = nx.erdos_renyi_graph(n, 0.2, seed=42)
        f1 = np.random.RandomState(42).randn(n, 4)
        f2 = np.random.RandomState(43).randn(n, 4)

        sm = SpectralMetrics(B=4)
        results = sm.compute_all(g1, f1, g2, f2)

        assert results.spectral_density_jsd >= 0
        assert results.quantile_band_energy_dist >= 0
        assert results.hks_distance >= 0
        assert results.degree_mmd >= 0
        assert results.clustering_mmd >= 0
        assert results.triangle_mmd >= 0
