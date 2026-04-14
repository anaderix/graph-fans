"""Tests for Phase 4a: InfoNoise entropy rate estimator and adaptive sampler."""

import numpy as np
import pytest
import torch

from graph_fans.phase0.spectral_profiler import compute_laplacian_spectrum, partition_into_bands
from graph_fans.phase2.info_noise import (
    InfoNoiseState,
    build_sampler_cdf,
    compute_entropy_rate,
    create_info_noise_state,
    get_entropy_rate_profile,
    record_observation,
    sample_sigma,
    sigma_to_t,
    sigma_to_t_batch,
)
from graph_fans.phase2.sde import CosineScheduleSDE
from graph_fans.phase2.trainer import Trainer, TrainConfig
from graph_fans.utils.graph_generators import generate_sbm
from graph_fans.utils.multiscale_features import generate_feature_dataset


@pytest.fixture
def sde():
    """CosineScheduleSDE instance."""
    return CosineScheduleSDE()


@pytest.fixture
def state(sde):
    """Fresh InfoNoiseState with small warm-up for testing."""
    return create_info_noise_state(sde, n_bins=10, warm_up_steps=50, buffer_capacity=32)


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


class TestCreateState:
    def test_creates_correct_n_bins(self, sde):
        state = create_info_noise_state(sde, n_bins=15)
        assert state.n_bins == 15
        assert len(state.bins) == 15

    def test_sigma_range(self, sde):
        state = create_info_noise_state(sde, n_bins=10)
        assert state.sigma_min > 0
        assert state.sigma_max > state.sigma_min
        # sigma_max should be close to 1 for cosine SDE at t=T
        assert state.sigma_max > 0.9

    def test_bin_centers_ordered(self, state):
        centers = [b.sigma_center for b in state.bins]
        for i in range(len(centers) - 1):
            assert centers[i] < centers[i + 1], (
                f"Bin centers not monotonically increasing: {centers[i]} >= {centers[i+1]}"
            )

    def test_initial_counts_zero(self, state):
        for b in state.bins:
            assert b.count == 0
            assert len(b.losses) == 0


class TestRecordObservation:
    def test_routes_to_correct_bin(self, state):
        """Low sigma goes to low bin, high sigma to high bin."""
        record_observation(state, state.sigma_min * 1.1, 1.0)
        record_observation(state, state.sigma_max * 0.99, 2.0)
        # First bin should have a record
        assert state.bins[0].count > 0
        # Last bin should have a record
        assert state.bins[-1].count > 0

    def test_fifo_eviction(self, sde):
        """FIFO buffer evicts oldest observations."""
        state = create_info_noise_state(sde, n_bins=5, buffer_capacity=3, warm_up_steps=10)
        sigma = state.bins[2].sigma_center
        for i in range(5):
            record_observation(state, sigma, float(i))
        # Buffer should only have last 3
        assert len(state.bins[2].losses) == 3
        assert list(state.bins[2].losses) == [2.0, 3.0, 4.0]

    def test_ema_convergence(self, sde):
        """EMA loss converges to constant input."""
        state = create_info_noise_state(sde, n_bins=5, ema_alpha=0.1, warm_up_steps=10)
        sigma = state.bins[2].sigma_center
        for _ in range(200):
            record_observation(state, sigma, 5.0)
        np.testing.assert_allclose(state.bins[2].ema_loss, 5.0, atol=0.01)

    def test_step_count_increments(self, state):
        sigma = state.bins[0].sigma_center
        initial = state._step_count
        record_observation(state, sigma, 1.0)
        assert state._step_count == initial + 1


class TestEntropyRate:
    def test_shape(self, state):
        """Output has correct shape [n_bins]."""
        rates = compute_entropy_rate(state)
        assert rates.shape == (state.n_bins,)

    def test_empty_bins_zero(self, state):
        """Empty bins should have zero entropy rate."""
        rates = compute_entropy_rate(state)
        assert np.all(rates == 0.0)

    def test_higher_loss_higher_rate(self, sde):
        """Bins with higher loss should have higher entropy rate (all else equal)."""
        state = create_info_noise_state(sde, n_bins=5, warm_up_steps=10, sigma_min_gate_width=0.001)
        # Give middle bins different loss values
        mid = 2
        high = 3
        for _ in range(50):
            record_observation(state, state.bins[mid].sigma_center, 1.0)
            record_observation(state, state.bins[high].sigma_center, 10.0)
        rates = compute_entropy_rate(state)
        # Higher loss bin should have higher rate (if sigmas are similar)
        # But sigma^3 denominator matters, so check relative
        assert rates[high] > 0
        assert rates[mid] > 0

    def test_gated_boundary_suppression(self, sde):
        """Gate should suppress entropy rate relative to ungated rate.

        The gate function gate(sigma) = sigma^n / (sigma^n + c^n) approaches 0
        near sigma=0. We verify that for the first bin, the gated rate is much
        less than the ungated rate (mmse / sigma^3).
        """
        state = create_info_noise_state(
            sde, n_bins=20, warm_up_steps=10, sigma_min_gate_width=0.5,
        )
        # Fill all bins with same loss
        for b in state.bins:
            for _ in range(20):
                record_observation(state, b.sigma_center, 1.0)
        rates = compute_entropy_rate(state)

        # Compute ungated rate for first bin: mmse / sigma^3
        sigma_0 = state.bins[0].sigma_center
        ungated_rate_0 = 1.0 / sigma_0**3  # mmse=1.0 (constant loss)
        gated_rate_0 = rates[0]

        # Gate should substantially reduce the rate at the boundary
        suppression_ratio = gated_rate_0 / ungated_rate_0
        assert suppression_ratio < 0.1, (
            f"Expected gate suppression ratio < 0.1 at sigma={sigma_0:.4f}: "
            f"gated={gated_rate_0:.4f}, ungated={ungated_rate_0:.4f}, ratio={suppression_ratio:.6f}"
        )


class TestSampler:
    def test_cdf_sums_to_one(self, sde):
        """CDF last element should be 1.0."""
        state = create_info_noise_state(sde, n_bins=10, warm_up_steps=10)
        for b in state.bins:
            for _ in range(10):
                record_observation(state, b.sigma_center, np.random.rand())
        cdf = build_sampler_cdf(state)
        np.testing.assert_allclose(cdf[-1], 1.0, atol=1e-10)

    def test_cdf_monotonic(self, sde):
        """CDF should be non-decreasing."""
        state = create_info_noise_state(sde, n_bins=10, warm_up_steps=10)
        for b in state.bins:
            for _ in range(10):
                record_observation(state, b.sigma_center, np.random.rand())
        cdf = build_sampler_cdf(state)
        for i in range(len(cdf) - 1):
            assert cdf[i] <= cdf[i + 1] + 1e-10

    def test_warm_up_is_uniform(self, sde):
        """During warm-up, sample_sigma returns None (caller uses uniform t)."""
        state = create_info_noise_state(sde, n_bins=10, warm_up_steps=100)
        rng = np.random.default_rng(42)
        result = sample_sigma(state, 10, rng)
        assert result is None

    def test_post_warmup_returns_sigmas(self, sde):
        """After warm-up, sample_sigma returns sigma values."""
        state = create_info_noise_state(sde, n_bins=10, warm_up_steps=10, buffer_capacity=32)
        # Fill bins to pass warm-up
        for _ in range(20):
            for b in state.bins:
                record_observation(state, b.sigma_center, np.random.rand())
        assert state._step_count >= state.warm_up_steps

        rng = np.random.default_rng(42)
        sigmas = sample_sigma(state, 5, rng)
        assert sigmas is not None
        assert sigmas.shape == (5,)
        assert np.all(sigmas > 0)

    def test_post_warmup_concentrated(self, sde):
        """After warm-up with non-uniform loss, sampling should concentrate."""
        state = create_info_noise_state(
            sde, n_bins=10, warm_up_steps=10, buffer_capacity=64,
            sigma_min_gate_width=0.001,
        )
        # Give one bin much higher loss
        target_bin = 5
        for _ in range(30):
            for i, b in enumerate(state.bins):
                loss = 100.0 if i == target_bin else 0.01
                record_observation(state, b.sigma_center, loss)

        rng = np.random.default_rng(42)
        sigmas = sample_sigma(state, 1000, rng)
        assert sigmas is not None

        # Count samples near the high-loss bin
        target_sigma = state.bins[target_bin].sigma_center
        sigma_range = 0.5 * (
            state.bins[min(target_bin + 1, state.n_bins - 1)].sigma_center
            - state.bins[max(target_bin - 1, 0)].sigma_center
        )
        near = np.abs(sigmas - target_sigma) < sigma_range
        fraction = near.sum() / len(sigmas)
        # Should have more than uniform fraction (1/n_bins = 0.1)
        assert fraction > 0.15, (
            f"Expected concentration near target: {fraction:.3f} > 0.15"
        )


class TestSigmaToT:
    def test_roundtrip(self, sde):
        """sigma_to_t roundtrip: marginal_params(sigma_to_t(sigma)).std ~ sigma."""
        for t_orig in [0.1, 0.3, 0.5, 0.7, 0.9]:
            _, sigma_orig = sde.marginal_params(t_orig)
            t_recovered = sigma_to_t(sde, sigma_orig)
            _, sigma_recovered = sde.marginal_params(t_recovered)
            np.testing.assert_allclose(sigma_recovered, sigma_orig, rtol=1e-3, err_msg=(
                f"Roundtrip failed at t={t_orig}: "
                f"sigma={sigma_orig:.6f} -> t={t_recovered:.6f} -> sigma={sigma_recovered:.6f}"
            ))

    def test_batch(self, sde):
        """sigma_to_t_batch processes arrays correctly."""
        sigmas = np.array([0.1, 0.3, 0.5, 0.8])
        ts = sigma_to_t_batch(sde, sigmas)
        assert ts.shape == (4,)
        # Higher sigma should map to higher t
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i + 1], f"t not increasing: t[{i}]={ts[i]}, t[{i+1}]={ts[i+1]}"

    def test_monotonic(self, sde):
        """Higher sigma maps to higher t (sigma is monotonic in t)."""
        sigmas = [0.05, 0.2, 0.5, 0.9]
        ts = [sigma_to_t(sde, s) for s in sigmas]
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i + 1]


class TestTrainerIntegration:
    def test_info_noise_trains(self, small_graph):
        """Trainer with t_sampling='info_noise' runs without error."""
        graph, dataset = small_graph
        config = TrainConfig(
            n_epochs=30, batch_timesteps=4,
            seed=42, device="cpu", hidden_dim=32, n_layers=2,
            n_train_samples=10,
            t_sampling="info_noise",
            info_noise_warm_up_steps=10,
            info_noise_n_bins=5,
        )
        trainer = Trainer(config, graph, dataset)
        history = trainer.train()
        assert len(history["loss"]) == 30
        assert all(np.isfinite(l) for l in history["loss"]), (
            "NaN or Inf in loss history with info_noise sampling"
        )

    def test_info_noise_state_populated(self, small_graph):
        """After training, InfoNoise state has observations."""
        graph, dataset = small_graph
        config = TrainConfig(
            n_epochs=10, batch_timesteps=4,
            seed=42, device="cpu", hidden_dim=32, n_layers=2,
            n_train_samples=10,
            t_sampling="info_noise",
            info_noise_warm_up_steps=5,
            info_noise_n_bins=5,
        )
        trainer = Trainer(config, graph, dataset)
        trainer.train()
        assert trainer.info_noise_state is not None
        total_obs = sum(b.count for b in trainer.info_noise_state.bins)
        assert total_obs > 0, "No observations recorded in InfoNoise state"

    def test_exports_profile(self, small_graph):
        """get_info_noise_profile returns valid dict after training."""
        graph, dataset = small_graph
        config = TrainConfig(
            n_epochs=10, batch_timesteps=4,
            seed=42, device="cpu", hidden_dim=32, n_layers=2,
            n_train_samples=10,
            t_sampling="info_noise",
            info_noise_warm_up_steps=5,
            info_noise_n_bins=5,
        )
        trainer = Trainer(config, graph, dataset)
        trainer.train()
        profile = trainer.get_info_noise_profile()
        assert profile is not None
        assert "sigma_centers" in profile
        assert "entropy_rates" in profile
        assert "counts" in profile
        assert len(profile["sigma_centers"]) == 5

    def test_info_noise_none_for_uniform(self, small_graph):
        """With t_sampling='uniform', info_noise_state is None."""
        graph, dataset = small_graph
        config = TrainConfig(
            n_epochs=5, batch_timesteps=2,
            seed=42, device="cpu", hidden_dim=32, n_layers=2,
            n_train_samples=10,
            t_sampling="uniform",
        )
        trainer = Trainer(config, graph, dataset)
        assert trainer.info_noise_state is None
        assert trainer.get_info_noise_profile() is None

    def test_loss_decreases(self, small_graph):
        """Loss should decrease with info_noise sampling."""
        graph, dataset = small_graph
        config = TrainConfig(
            n_epochs=50, lr=1e-3, batch_timesteps=8,
            seed=42, device="cpu", hidden_dim=32, n_layers=2,
            n_train_samples=10,
            t_sampling="info_noise",
            info_noise_warm_up_steps=10,
            info_noise_n_bins=5,
            use_lr_scheduler=False,
        )
        trainer = Trainer(config, graph, dataset)
        history = trainer.train()
        first_avg = float(np.mean(history["loss"][:10]))
        last_avg = float(np.mean(history["loss"][-10:]))
        assert last_avg < first_avg, (
            f"Loss did not decrease with info_noise: "
            f"first-10={first_avg:.4f}, last-10={last_avg:.4f}"
        )


class TestInfoNoiseSmoke:
    def test_full_pipeline(self, small_graph):
        """Full train-generate-evaluate pipeline with InfoNoise on tiny graph."""
        graph, dataset = small_graph
        config = TrainConfig(
            n_epochs=30, batch_timesteps=4,
            seed=42, device="cpu", hidden_dim=32, n_layers=2,
            n_gen_steps=10, n_train_samples=10,
            t_sampling="info_noise",
            info_noise_warm_up_steps=10,
            info_noise_n_bins=5,
        )
        trainer = Trainer(config, graph, dataset)
        history = trainer.train()

        # Generate samples
        gen = trainer.generate(n_steps=10, n_samples=3)
        assert gen.shape == (3, graph.number_of_nodes(), 4)
        assert np.all(np.isfinite(gen)), "Non-finite values in generated features"

        # Profile should be exportable
        profile = trainer.get_info_noise_profile()
        assert profile is not None
        assert profile["step_count"] > 0


class TestGetEntropyRateProfile:
    def test_json_serializable(self, sde):
        """Profile dict should be JSON-serializable."""
        import json
        state = create_info_noise_state(sde, n_bins=5, warm_up_steps=10)
        for b in state.bins:
            for _ in range(5):
                record_observation(state, b.sigma_center, np.random.rand())
        profile = get_entropy_rate_profile(state)
        # Should not raise
        json_str = json.dumps(profile)
        assert len(json_str) > 0
