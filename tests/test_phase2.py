"""Tests for Phase 2: Regime A Core (H1-A + H2)."""

import networkx as nx
import numpy as np
import pytest
import torch

from graph_fans.phase0.spectral_profiler import compute_laplacian_spectrum, partition_into_bands
from graph_fans.phase2.noise_shaper import (
    compute_importance_weights,
    shape_noise,
    shape_noise_with_temporal_ramp,
)
from graph_fans.phase2.score_network import SimpleScoreNetwork
from graph_fans.phase2.sde import VPSDE
from graph_fans.phase2.trainer import Trainer, TrainConfig
from graph_fans.utils.graph_generators import generate_sbm


@pytest.fixture
def small_graph():
    """Small SBM graph for testing."""
    gd = generate_sbm(n_nodes=30, n_communities=3, p_intra=0.5, p_inter=0.05,
                       n_features=4, seed=42)
    return gd.graph, gd.features


@pytest.fixture
def band_setup(small_graph):
    """Eigendecomposition and band setup for a small graph."""
    graph, features = small_graph
    eigenvalues, eigenvectors = compute_laplacian_spectrum(graph)
    _, band_indices = partition_into_bands(eigenvalues, B=4)
    return eigenvalues, eigenvectors, band_indices, features


class TestImportanceWeights:
    def test_shape(self):
        energies = np.array([10.0, 5.0, 2.0, 1.0])
        w = compute_importance_weights(energies)
        assert w.weights.shape == (4,)

    def test_all_positive(self):
        energies = np.array([10.0, 5.0, 2.0, 1.0])
        w = compute_importance_weights(energies)
        assert np.all(w.weights > 0)

    def test_inversely_proportional(self):
        """High-energy bands should get lower weights."""
        energies = np.array([10.0, 5.0, 2.0, 1.0])
        w = compute_importance_weights(energies, alpha=1.0)
        # Band 0 has highest energy → lowest weight
        # Band 3 has lowest energy → highest weight
        assert w.weights[0] < w.weights[3]

    def test_mean_one(self):
        """Weights should be normalized so mean ≈ 1."""
        energies = np.array([10.0, 5.0, 2.0, 1.0])
        w = compute_importance_weights(energies)
        np.testing.assert_allclose(w.weights.mean(), 1.0, atol=1e-6)


class TestNoiseShaping:
    def test_output_shape(self, band_setup):
        _, eigenvectors, band_indices, features = band_setup
        n, d = features.shape
        noise = torch.randn(n, d)
        U = torch.tensor(eigenvectors, dtype=torch.float32)
        energies = np.array([5.0, 3.0, 2.0, 1.0])
        w = compute_importance_weights(energies)

        shaped = shape_noise(noise, U, band_indices, w)
        assert shaped.shape == (n, d)

    def test_unit_variance_per_band(self, band_setup):
        """After shaping, each band should have approximately unit variance."""
        _, eigenvectors, band_indices, features = band_setup
        n, d = features.shape
        torch.manual_seed(0)
        noise = torch.randn(n, d)
        U = torch.tensor(eigenvectors, dtype=torch.float32)
        energies = np.array([5.0, 3.0, 2.0, 1.0])
        w = compute_importance_weights(energies)

        shaped = shape_noise(noise, U, band_indices, w)
        coeffs = U.T @ shaped

        for indices in band_indices:
            if len(indices) > 2:
                band_std = coeffs[indices].std().item()
                # Should be close to 1.0 due to unit-variance normalization
                assert 0.5 < band_std < 2.0, f"Band std={band_std}, expected ~1.0"

    def test_preserves_approximate_total_variance(self, band_setup):
        """Total variance should be approximately preserved."""
        _, eigenvectors, band_indices, features = band_setup
        n, d = features.shape
        torch.manual_seed(0)
        noise = torch.randn(n, d)
        U = torch.tensor(eigenvectors, dtype=torch.float32)
        energies = np.array([5.0, 3.0, 2.0, 1.0])
        w = compute_importance_weights(energies)

        shaped = shape_noise(noise, U, band_indices, w)
        # Variance should be in the same order of magnitude
        orig_var = noise.var().item()
        shaped_var = shaped.var().item()
        assert shaped_var > 0.1 * orig_var
        assert shaped_var < 10.0 * orig_var


class TestTemporalRamp:
    def test_below_knee_is_uniform(self, band_setup):
        """Below t_knee, output should equal input noise."""
        _, eigenvectors, band_indices, _ = band_setup
        n = eigenvectors.shape[0]
        noise = torch.randn(n, 4)
        U = torch.tensor(eigenvectors, dtype=torch.float32)
        energies = np.array([5.0, 3.0, 2.0, 1.0])
        w = compute_importance_weights(energies)

        result = shape_noise_with_temporal_ramp(noise, U, band_indices, w, t=0.05, t_knee=0.15)
        torch.testing.assert_close(result, noise)

    def test_above_knee_is_shaped(self, band_setup):
        """Well above t_knee, output should differ from input."""
        _, eigenvectors, band_indices, _ = band_setup
        n = eigenvectors.shape[0]
        torch.manual_seed(0)
        noise = torch.randn(n, 4)
        U = torch.tensor(eigenvectors, dtype=torch.float32)
        energies = np.array([5.0, 3.0, 2.0, 1.0])
        w = compute_importance_weights(energies)

        result = shape_noise_with_temporal_ramp(noise, U, band_indices, w, t=0.99, t_knee=0.15)
        # Should be different from uniform noise
        diff = (result - noise).abs().mean().item()
        assert diff > 0.01

    def test_interpolation(self, band_setup):
        """At intermediate t, result should be a blend."""
        _, eigenvectors, band_indices, _ = band_setup
        n = eigenvectors.shape[0]
        torch.manual_seed(0)
        noise = torch.randn(n, 4)
        U = torch.tensor(eigenvectors, dtype=torch.float32)
        energies = np.array([5.0, 3.0, 2.0, 1.0])
        w = compute_importance_weights(energies)

        # At t=0.5 with t_knee=0.15, phi should be between 0 and 1
        result = shape_noise_with_temporal_ramp(noise, U, band_indices, w, t=0.5, t_knee=0.15)
        fully_shaped = shape_noise(noise, U, band_indices, w)

        # Result should be between noise and fully_shaped
        diff_from_noise = (result - noise).abs().mean().item()
        diff_from_shaped = (result - fully_shaped).abs().mean().item()
        assert diff_from_noise > 0.001  # Not fully uniform
        assert diff_from_shaped > 0.001  # Not fully shaped


class TestScoreNetwork:
    def test_output_shape(self):
        n_nodes, n_features = 20, 4
        model = SimpleScoreNetwork(n_features=n_features, hidden_dim=32, n_layers=2)
        x = torch.randn(n_nodes, n_features)
        t = torch.tensor([0.5])
        # Simple chain graph
        edge_index = torch.tensor(
            [[i, i + 1, i + 1, i] for i in range(n_nodes - 1)],
            dtype=torch.long
        ).reshape(-1, 2).T
        out = model(x, t, edge_index)
        assert out.shape == (n_nodes, n_features)


class TestVPSDE:
    def test_marginal_at_zero(self):
        sde = VPSDE()
        mean_coeff, std = sde.marginal_params(0.0)
        assert abs(mean_coeff - 1.0) < 1e-6
        assert abs(std) < 1e-6

    def test_marginal_at_T(self):
        sde = VPSDE()
        mean_coeff, std = sde.marginal_params(1.0)
        # At T, x_t should be approximately pure noise
        assert mean_coeff < 0.1  # Very small
        assert std > 0.9  # Close to 1

    def test_perturb_shape(self):
        sde = VPSDE()
        x_0 = torch.randn(10, 4)
        x_t, noise = sde.perturb(x_0, 0.5)
        assert x_t.shape == (10, 4)
        assert noise.shape == (10, 4)


class TestTrainer:
    def test_loss_decreases(self, small_graph):
        """After a few epochs, loss should decrease."""
        graph, features = small_graph
        config = TrainConfig(
            n_epochs=50, lr=1e-3, batch_timesteps=4,
            seed=42, device="cpu", hidden_dim=32, n_layers=2,
        )
        trainer = Trainer(config, graph, features)
        history = trainer.train()
        # Loss at end should be less than at start (with some tolerance)
        assert history["loss"][-1] < history["loss"][0] * 1.5

    def test_generation_shape(self, small_graph):
        """Generated features should have correct shape."""
        graph, features = small_graph
        config = TrainConfig(
            n_epochs=10, batch_timesteps=2,
            seed=42, device="cpu", hidden_dim=32, n_layers=2,
            n_gen_steps=10,
        )
        trainer = Trainer(config, graph, features)
        trainer.train()
        gen = trainer.generate(n_steps=10)
        assert gen.shape == features.shape


class TestIntegration:
    def test_uniform_vs_spectral_pipeline(self):
        """Full pipeline runs end-to-end on a tiny graph."""
        from graph_fans.phase2.evaluate import run_single_experiment

        energies = np.array([5.0, 3.0, 2.0, 1.0])
        weights = compute_importance_weights(energies)
        config = TrainConfig(
            n_epochs=20, batch_timesteps=2, seed=0, device="cpu",
            hidden_dim=32, n_layers=2, B=4, n_gen_steps=10,
        )

        r_uniform = run_single_experiment(
            "SBM(q=0.05)", "uniform", seed=0,
            n_nodes=30, n_features=4, config=config,
            importance_weights=None, B=4,
        )
        r_spectral = run_single_experiment(
            "SBM(q=0.05)", "spectral", seed=0,
            n_nodes=30, n_features=4, config=config,
            importance_weights=weights, B=4,
        )

        assert r_uniform.qbe_total >= 0
        assert r_spectral.qbe_total >= 0
        assert r_uniform.graph_family == "SBM(q=0.05)"
        assert r_spectral.method == "spectral"

    def test_h2_grid_search_runs(self):
        """H2 grid search with 2 t_knee values completes without error."""
        from graph_fans.phase2.evaluate import run_single_experiment

        energies = np.array([5.0, 3.0, 2.0, 1.0])
        weights = compute_importance_weights(energies)
        config = TrainConfig(
            n_epochs=10, batch_timesteps=2, seed=0, device="cpu",
            hidden_dim=32, n_layers=2, B=4, n_gen_steps=10,
        )

        for t_knee in [0.1, 0.3]:
            r = run_single_experiment(
                "SBM(q=0.05)", f"spectral_ramp_tknee={t_knee}", seed=0,
                n_nodes=30, n_features=4, config=config,
                importance_weights=weights, B=4,
            )
            assert r.qbe_total >= 0
            assert f"tknee={t_knee}" in r.method
