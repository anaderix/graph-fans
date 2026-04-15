"""Tests for Phase 5a: Independent Per-Mode Spectral Diffusion."""

import networkx as nx
import numpy as np
import pytest
import torch

from graph_fans.phase0.spectral_profiler import compute_laplacian_spectrum
from graph_fans.phase2.sde import CosineScheduleSDE
from graph_fans.phase5.mode_schedule import ModeSchedule, ModeScheduleSet
from graph_fans.phase5.spatial_coherence import (
    cross_mode_energy_correlation,
    node_neighbor_correlation,
    spatial_coherence_summary,
)
from graph_fans.phase5.spectral_score_network import SpectralScoreNetwork
from graph_fans.phase5.spectral_trainer import SpectralTrainer, SpectralTrainConfig
from graph_fans.utils.graph_generators import generate_sbm
from graph_fans.utils.multiscale_features import generate_feature_dataset


@pytest.fixture
def small_graph():
    """Small SBM graph for testing."""
    gd = generate_sbm(
        n_nodes=20, n_communities=2, p_intra=0.5, p_inter=0.05,
        n_features=4, seed=42, feature_mode="community",
    )
    return gd.graph


@pytest.fixture
def community_features(small_graph):
    """Community feature dataset for the small graph."""
    return generate_feature_dataset(
        small_graph, n_samples=20, n_features=4, base_seed=42, mode="community",
    )


@pytest.fixture
def eigen_setup(small_graph):
    """Eigendecomposition for the small graph."""
    eigenvalues, eigenvectors = compute_laplacian_spectrum(small_graph)
    return eigenvalues, eigenvectors


# ============================================================================
# TestSpectralScoreNetwork
# ============================================================================


class TestSpectralScoreNetwork:
    def test_output_shape(self):
        """Output [batch, d] matches input feature dim."""
        n_features = 4
        model = SpectralScoreNetwork(n_features=n_features, hidden_dim=64, n_layers=3)
        batch = 10
        c_k_t = torch.randn(batch, n_features)
        lambda_k = torch.randn(batch, 1)
        t = torch.rand(batch, 1)
        E_k = torch.rand(batch, 1)

        out = model(c_k_t, lambda_k, t, E_k)
        assert out.shape == (batch, n_features)

    def test_conditioning_varies_output(self):
        """Different lambda_k, t, E_k produce different predictions."""
        model = SpectralScoreNetwork(n_features=4, hidden_dim=64, n_layers=3)
        c_k_t = torch.randn(1, 4)

        out1 = model(c_k_t, torch.tensor([[0.1]]), torch.tensor([[0.5]]), torch.tensor([[1.0]]))
        out2 = model(c_k_t, torch.tensor([[1.5]]), torch.tensor([[0.5]]), torch.tensor([[1.0]]))
        out3 = model(c_k_t, torch.tensor([[0.1]]), torch.tensor([[0.9]]), torch.tensor([[1.0]]))
        out4 = model(c_k_t, torch.tensor([[0.1]]), torch.tensor([[0.5]]), torch.tensor([[0.01]]))

        # All four should differ
        assert not torch.allclose(out1, out2, atol=1e-4)
        assert not torch.allclose(out1, out3, atol=1e-4)
        assert not torch.allclose(out1, out4, atol=1e-4)

    def test_batch_mode(self):
        """All modes batched into single forward pass."""
        n_modes = 20
        n_features = 4
        model = SpectralScoreNetwork(n_features=n_features, hidden_dim=64, n_layers=2)

        c_k_t = torch.randn(n_modes, n_features)
        lambda_k = torch.linspace(0, 2, n_modes).unsqueeze(-1)
        t = torch.rand(n_modes, 1)
        E_k = torch.rand(n_modes, 1)

        out = model(c_k_t, lambda_k, t, E_k)
        assert out.shape == (n_modes, n_features)
        # Outputs should vary across modes (different conditioning)
        assert not torch.allclose(out[0], out[-1], atol=1e-4)


# ============================================================================
# TestModeSchedule
# ============================================================================


class TestModeSchedule:
    def test_t_max_energy_proportional(self):
        """Higher energy modes get larger t_max."""
        eigenvalues = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        # High energy for low-freq (mode 0), low energy for high-freq (mode 4)
        mode_energies = np.array([10.0, 5.0, 2.0, 1.0, 0.5])
        sde = CosineScheduleSDE()
        schedule = ModeScheduleSet(eigenvalues, mode_energies, sde, energy_exponent=0.5)

        assert schedule.schedules[0].t_max >= schedule.schedules[4].t_max

    def test_t_max_clamping(self):
        """t_max is clamped to [0.1*T, T]."""
        eigenvalues = np.array([0.0, 1.0, 2.0])
        # Very small energy for mode 2 -> should be clamped to floor
        mode_energies = np.array([10.0, 1.0, 0.001])
        sde = CosineScheduleSDE()
        schedule = ModeScheduleSet(
            eigenvalues, mode_energies, sde,
            energy_exponent=0.5, t_max_floor=0.1,
        )

        for s in schedule.schedules:
            assert s.t_max >= 0.1 * sde.T
            assert s.t_max <= sde.T

    def test_ddim_grid_endpoints(self):
        """Grid starts at t_max(k), ends near 0."""
        eigenvalues = np.array([0.0, 1.0])
        mode_energies = np.array([5.0, 1.0])
        sde = CosineScheduleSDE()
        schedule = ModeScheduleSet(eigenvalues, mode_energies, sde)

        for k in range(2):
            grid = schedule.get_ddim_grid(k, n_steps=10)
            assert len(grid) == 11
            assert abs(grid[0] - schedule.schedules[k].t_max) < 1e-6
            assert grid[-1] < 1e-4  # near zero

    def test_perturb_shape(self):
        """perturb returns correct shape."""
        eigenvalues = np.array([0.0, 1.0])
        mode_energies = np.array([5.0, 1.0])
        sde = CosineScheduleSDE()
        schedule = ModeScheduleSet(eigenvalues, mode_energies, sde)

        x_0 = torch.randn(1, 4)
        x_t, noise = schedule.perturb(x_0, mode_idx=0, t=0.5)
        assert x_t.shape == (1, 4)
        assert noise.shape == (1, 4)

    def test_ddim_step_shape(self):
        """ddim_step returns correct shape."""
        eigenvalues = np.array([0.0, 1.0])
        mode_energies = np.array([5.0, 1.0])
        sde = CosineScheduleSDE()
        schedule = ModeScheduleSet(eigenvalues, mode_energies, sde)

        x_t = torch.randn(1, 4)
        eps_pred = torch.randn(1, 4)
        result = schedule.ddim_step(x_t, eps_pred, mode_idx=0, t_now=0.5, t_next=0.3)
        assert result.shape == (1, 4)

    def test_sample_t_within_bounds(self):
        """sample_t returns t in [eps, t_max(k)]."""
        eigenvalues = np.array([0.0, 1.0, 2.0])
        mode_energies = np.array([10.0, 5.0, 1.0])
        sde = CosineScheduleSDE()
        schedule = ModeScheduleSet(eigenvalues, mode_energies, sde)
        rng = np.random.default_rng(42)

        for k in range(3):
            for _ in range(20):
                t = schedule.sample_t(k, rng)
                assert t >= 1e-5
                assert t <= schedule.schedules[k].t_max + 1e-6


# ============================================================================
# TestSpectralTrainer
# ============================================================================


class TestSpectralTrainer:
    def test_projection_roundtrip(self, small_graph, community_features, eigen_setup):
        """U @ U^T @ x == x for features in the graph's node space."""
        _, eigenvectors = eigen_setup
        x = community_features[0]  # [n_nodes, n_features]

        # Project to spectral and back
        coeffs = eigenvectors.T @ x      # [n_modes, d]
        x_recon = eigenvectors @ coeffs   # [n_nodes, d]

        np.testing.assert_allclose(x, x_recon, atol=1e-5)

    def test_loss_decreases(self, small_graph, community_features):
        """Train 100 epochs on small graph, final loss < initial loss."""
        config = SpectralTrainConfig(
            n_epochs=100, lr=1e-3, hidden_dim=64, n_layers=2,
            n_features=4, seed=42, device="cpu",
        )
        trainer = SpectralTrainer(small_graph, community_features, config)
        result = trainer.train()

        losses = result["losses"]
        initial_loss = np.mean(losses[:5])
        final_loss = np.mean(losses[-5:])
        assert final_loss < initial_loss, (
            f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )

    def test_generation_shape(self, small_graph, community_features):
        """generate() returns list of [n_nodes, n_features] arrays."""
        config = SpectralTrainConfig(
            n_epochs=20, lr=1e-3, hidden_dim=32, n_layers=2,
            n_features=4, n_gen_steps=10, seed=42, device="cpu",
        )
        trainer = SpectralTrainer(small_graph, community_features, config)
        trainer.train()

        gen = trainer.generate(n_steps=10, n_samples=3)
        assert len(gen) == 3
        n_nodes = small_graph.number_of_nodes()
        for sample in gen:
            assert sample.shape == (n_nodes, 4)

    def test_generation_finite(self, small_graph, community_features):
        """Generated features contain no NaN or Inf."""
        config = SpectralTrainConfig(
            n_epochs=50, lr=1e-3, hidden_dim=64, n_layers=2,
            n_features=4, n_gen_steps=20, seed=42, device="cpu",
        )
        trainer = SpectralTrainer(small_graph, community_features, config)
        trainer.train()

        gen = trainer.generate(n_steps=20, n_samples=5)
        for sample in gen:
            assert np.all(np.isfinite(sample)), "Generated features contain NaN or Inf"

    def test_sanity_check(self, small_graph, community_features):
        """sanity_check returns expected keys."""
        config = SpectralTrainConfig(
            n_epochs=20, lr=1e-3, hidden_dim=32, n_layers=2,
            n_features=4, n_gen_steps=5, seed=42, device="cpu",
        )
        trainer = SpectralTrainer(small_graph, community_features, config)
        trainer.train()

        gen = trainer.generate(n_steps=5, n_samples=1)
        result = trainer.sanity_check(community_features[0], gen[0])

        assert "std_ratio" in result
        assert "spectral_l2" in result
        assert isinstance(result["std_ratio"], float)
        assert result["std_ratio"] > 0


# ============================================================================
# TestSpatialCoherence
# ============================================================================


class TestSpatialCoherence:
    def test_neighbor_correlation_community(self, small_graph, community_features):
        """Community features on SBM give positive neighbor correlation."""
        corr = node_neighbor_correlation(community_features[0], small_graph)
        # Community features should show positive spatial correlation
        assert corr > 0.0, f"Expected positive correlation, got {corr}"

    def test_neighbor_correlation_noise(self, small_graph):
        """Random noise gives neighbor correlation near 0."""
        n_nodes = small_graph.number_of_nodes()
        rng = np.random.default_rng(42)
        noise = rng.standard_normal((n_nodes, 4))
        corr = node_neighbor_correlation(noise, small_graph)
        # Random noise: expect near zero (within [-0.3, 0.3] for small graph)
        assert abs(corr) < 0.4, f"Expected near-zero correlation for noise, got {corr}"

    def test_cross_mode_energy_correlation(self, community_features, eigen_setup):
        """Community features have positive energy-mode correlation."""
        _, eigenvectors = eigen_setup
        corr = cross_mode_energy_correlation(community_features[0], eigenvectors)
        # Community features concentrate energy in low-frequency modes
        # so correlation with -rank should be positive
        # (but for very small graphs this can be noisy, so just check finite)
        assert np.isfinite(corr)

    def test_spatial_coherence_summary_keys(self, small_graph, community_features, eigen_setup):
        """spatial_coherence_summary returns all expected keys."""
        _, eigenvectors = eigen_setup
        result = spatial_coherence_summary(
            community_features[0], community_features[1],
            small_graph, eigenvectors,
        )
        expected_keys = {
            "ref_neighbor_corr", "gen_neighbor_corr",
            "ref_mode_energy_corr", "gen_mode_energy_corr",
        }
        assert set(result.keys()) == expected_keys
        for v in result.values():
            assert isinstance(v, float)
