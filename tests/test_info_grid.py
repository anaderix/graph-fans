"""Tests for Phase 4b: InfoGrid non-uniform DDIM step spacing."""

import numpy as np
import pytest
import torch

from graph_fans.phase2.info_grid import build_info_grid, build_uniform_grid
from graph_fans.phase2.info_noise import (
    create_info_noise_state,
    get_entropy_rate_profile,
    record_observation,
)
from graph_fans.phase2.sde import CosineScheduleSDE
from graph_fans.phase2.trainer import Trainer, TrainConfig
from graph_fans.utils.graph_generators import generate_sbm
from graph_fans.utils.multiscale_features import generate_feature_dataset


@pytest.fixture
def sde():
    return CosineScheduleSDE()


@pytest.fixture
def entropy_rate_profile(sde):
    """Build a synthetic entropy rate profile with a peak in the middle."""
    state = create_info_noise_state(sde, n_bins=20, warm_up_steps=10, sigma_min_gate_width=0.01)
    # Create a profile with a peak around the middle bins
    for i, b in enumerate(state.bins):
        # Triangular peak centered at bin 10
        loss = max(0.1, 10.0 - abs(i - 10) * 1.0)
        for _ in range(30):
            record_observation(state, b.sigma_center, loss)
    return get_entropy_rate_profile(state)


@pytest.fixture
def small_graph():
    """Small SBM graph with feature dataset for testing."""
    gd = generate_sbm(
        n_nodes=20, n_communities=2, p_intra=0.5, p_inter=0.05,
        n_features=4, seed=42, feature_mode="smooth",
    )
    dataset = generate_feature_dataset(
        gd.graph, n_samples=10, n_features=4,
        base_seed=42, mode="smooth",
    )
    return gd.graph, dataset


class TestBuildInfoGrid:
    def test_correct_length(self, entropy_rate_profile, sde):
        """Grid should have n_steps + 1 elements."""
        grid = build_info_grid(entropy_rate_profile, sde, n_steps=50)
        assert len(grid) == 51

    def test_starts_at_T(self, entropy_rate_profile, sde):
        """Grid should start at sde.T."""
        grid = build_info_grid(entropy_rate_profile, sde, n_steps=50)
        np.testing.assert_allclose(grid[0], sde.T, atol=1e-6)

    def test_ends_at_zero(self, entropy_rate_profile, sde):
        """Grid should end at 0."""
        grid = build_info_grid(entropy_rate_profile, sde, n_steps=50)
        np.testing.assert_allclose(grid[-1], 0.0, atol=1e-6)

    def test_monotonically_decreasing(self, entropy_rate_profile, sde):
        """Grid should be monotonically decreasing (T -> 0)."""
        grid = build_info_grid(entropy_rate_profile, sde, n_steps=100)
        for i in range(len(grid) - 1):
            assert grid[i] >= grid[i + 1], (
                f"Grid not decreasing at index {i}: {grid[i]} < {grid[i+1]}"
            )

    def test_concentrates_in_high_entropy_region(self, entropy_rate_profile, sde):
        """More steps should be allocated to the high-entropy region.

        We check that step density in the middle (where we placed the
        entropy peak) is higher than at the edges.
        """
        grid = build_info_grid(entropy_rate_profile, sde, n_steps=200)
        diffs = np.abs(np.diff(grid))

        # Divide grid into thirds
        n = len(diffs)
        third = n // 3
        edge_mean_dt = 0.5 * (diffs[:third].mean() + diffs[-third:].mean())
        middle_mean_dt = diffs[third:2*third].mean()

        # In high-entropy region, steps are closer together (smaller dt)
        # So middle_mean_dt should be smaller than edge_mean_dt
        assert middle_mean_dt < edge_mean_dt, (
            f"Expected concentration in middle: middle_dt={middle_mean_dt:.4f} "
            f"should be < edge_dt={edge_mean_dt:.4f}"
        )


class TestBuildUniformGrid:
    def test_correct_length(self, sde):
        grid = build_uniform_grid(sde, n_steps=100)
        assert len(grid) == 101

    def test_starts_at_T_ends_at_zero(self, sde):
        grid = build_uniform_grid(sde, n_steps=50)
        np.testing.assert_allclose(grid[0], sde.T, atol=1e-10)
        np.testing.assert_allclose(grid[-1], 0.0, atol=1e-10)

    def test_uniform_spacing(self, sde):
        """Uniform grid should have equal spacing."""
        grid = build_uniform_grid(sde, n_steps=100)
        diffs = np.diff(grid)
        np.testing.assert_allclose(diffs, diffs[0], atol=1e-10)


class TestGenerateWithGrid:
    def test_runs_with_uniform_grid(self, small_graph):
        """generate_with_grid with uniform grid should produce valid output."""
        graph, dataset = small_graph
        config = TrainConfig(
            n_epochs=10, batch_timesteps=2,
            seed=42, device="cpu", hidden_dim=32, n_layers=2,
            n_gen_steps=10, n_train_samples=10,
        )
        trainer = Trainer(config, graph, dataset)
        trainer.train()

        sde = trainer.sde
        ts = build_uniform_grid(sde, n_steps=10)
        gen = trainer.generate_with_grid(ts, n_samples=2)
        assert gen.shape == (2, graph.number_of_nodes(), 4)
        assert np.all(np.isfinite(gen))

    def test_uniform_grid_matches_default_generate(self, small_graph):
        """generate_with_grid(uniform_grid) should match generate() exactly."""
        graph, dataset = small_graph
        config = TrainConfig(
            n_epochs=5, batch_timesteps=2,
            seed=42, device="cpu", hidden_dim=32, n_layers=2,
            n_gen_steps=10, n_train_samples=10,
        )
        trainer = Trainer(config, graph, dataset)
        trainer.train()

        sde = trainer.sde
        ts = build_uniform_grid(sde, n_steps=10)

        torch.manual_seed(0)
        gen_default = trainer.generate(n_steps=10, n_samples=1)

        torch.manual_seed(0)
        gen_grid = trainer.generate_with_grid(ts, n_samples=1)

        np.testing.assert_allclose(gen_default, gen_grid, atol=1e-5,
            err_msg="generate_with_grid(uniform) should match default generate()")

    def test_runs_with_info_grid(self, small_graph, entropy_rate_profile):
        """generate_with_grid with InfoGrid should produce valid output."""
        graph, dataset = small_graph
        config = TrainConfig(
            n_epochs=10, batch_timesteps=2,
            seed=42, device="cpu", hidden_dim=32, n_layers=2,
            n_gen_steps=10, n_train_samples=10,
        )
        trainer = Trainer(config, graph, dataset)
        trainer.train()

        sde = trainer.sde
        ts = build_info_grid(entropy_rate_profile, sde, n_steps=10)
        gen = trainer.generate_with_grid(ts, n_samples=2)
        assert gen.shape == (2, graph.number_of_nodes(), 4)
        assert np.all(np.isfinite(gen))
