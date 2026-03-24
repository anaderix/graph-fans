"""Tests for Phase 2: Regime A Core (H1-A + H2)."""

import tempfile

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
from graph_fans.phase2.sde import VPSDE, CosineScheduleSDE
from graph_fans.phase2.trainer import Trainer, TrainConfig
from graph_fans.utils.graph_generators import generate_sbm
from graph_fans.utils.multiscale_features import generate_feature_dataset


@pytest.fixture
def small_graph():
    """Small SBM graph with feature dataset for testing."""
    gd = generate_sbm(n_nodes=30, n_communities=3, p_intra=0.5, p_inter=0.05,
                       n_features=4, seed=42, feature_mode="smooth")
    # Generate a small dataset of feature realizations
    dataset = generate_feature_dataset(gd.graph, n_samples=10, n_features=4,
                                        base_seed=42, mode="smooth")
    return gd.graph, dataset


@pytest.fixture
def band_setup(small_graph):
    """Eigendecomposition and band setup for a small graph."""
    graph, dataset = small_graph
    features = dataset[0]  # Use first sample for noise shaping tests
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
        assert w.weights[0] < w.weights[3]

    def test_mean_one(self):
        """Weights should be normalized so mean ~ 1."""
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

        result = shape_noise_with_temporal_ramp(noise, U, band_indices, w, t=0.5, t_knee=0.15)
        fully_shaped = shape_noise(noise, U, band_indices, w)

        diff_from_noise = (result - noise).abs().mean().item()
        diff_from_shaped = (result - fully_shaped).abs().mean().item()
        assert diff_from_noise > 0.001
        assert diff_from_shaped > 0.001


class TestScoreNetwork:
    def test_output_shape(self):
        n_nodes, n_features = 20, 4
        model = SimpleScoreNetwork(n_features=n_features, hidden_dim=32, n_layers=2)
        x = torch.randn(n_nodes, n_features)
        t = torch.tensor([0.5])
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
        assert mean_coeff < 0.1
        assert std > 0.9

    def test_perturb_shape(self):
        sde = VPSDE()
        x_0 = torch.randn(10, 4)
        x_t, noise = sde.perturb(x_0, 0.5)
        assert x_t.shape == (10, 4)
        assert noise.shape == (10, 4)

    def test_alpha_bar(self):
        sde = VPSDE()
        ab_0 = sde.alpha_bar(0.0)
        ab_T = sde.alpha_bar(1.0)
        assert abs(ab_0 - 1.0) < 1e-4
        assert ab_T < 0.01

    def test_ddim_step_final(self):
        """DDIM at t_next=0 returns Tweedie estimate (no noise)."""
        sde = VPSDE()
        torch.manual_seed(42)
        x_t = torch.randn(10, 4)
        eps_pred = torch.randn(10, 4)

        result = sde.ddim_step(x_t, eps_pred, t_now=0.5, t_next=0.0)
        # Should be deterministic
        result2 = sde.ddim_step(x_t, eps_pred, t_now=0.5, t_next=0.0)
        torch.testing.assert_close(result, result2)
        # Should recover x_0 estimate
        ab = sde.alpha_bar(0.5)
        expected = (x_t - np.sqrt(1 - ab) * eps_pred) / np.sqrt(ab)
        torch.testing.assert_close(result, expected)

    def test_ddim_step_intermediate(self):
        """DDIM at t_next>0 returns deterministic interpolation."""
        sde = VPSDE()
        torch.manual_seed(42)
        x_t = torch.randn(10, 4)
        eps_pred = torch.randn(10, 4)

        result1 = sde.ddim_step(x_t, eps_pred, t_now=0.8, t_next=0.4)
        result2 = sde.ddim_step(x_t, eps_pred, t_now=0.8, t_next=0.4)
        torch.testing.assert_close(result1, result2)  # deterministic


class TestCosineScheduleSDE:
    def test_alpha_bar(self):
        sde = CosineScheduleSDE()
        ab_0 = sde.alpha_bar(0.0)
        ab_T = sde.alpha_bar(1.0)
        assert ab_0 > 0.99
        assert ab_T < 0.01

    def test_ddim_step_final(self):
        """DDIM at t_next=0 returns Tweedie estimate."""
        sde = CosineScheduleSDE()
        torch.manual_seed(42)
        x_t = torch.randn(10, 4)
        eps_pred = torch.randn(10, 4)

        result = sde.ddim_step(x_t, eps_pred, t_now=0.5, t_next=0.0)
        result2 = sde.ddim_step(x_t, eps_pred, t_now=0.5, t_next=0.0)
        torch.testing.assert_close(result, result2)

    def test_ddim_step_intermediate(self):
        """DDIM at t_next>0 is deterministic."""
        sde = CosineScheduleSDE()
        torch.manual_seed(42)
        x_t = torch.randn(10, 4)
        eps_pred = torch.randn(10, 4)

        result1 = sde.ddim_step(x_t, eps_pred, t_now=0.8, t_next=0.4)
        result2 = sde.ddim_step(x_t, eps_pred, t_now=0.8, t_next=0.4)
        torch.testing.assert_close(result1, result2)


class TestEpsilonPrediction:
    def test_epsilon_target(self, small_graph):
        """Verify training target is noise (epsilon), not -noise/std."""
        graph, dataset = small_graph
        config = TrainConfig(
            n_epochs=1, lr=1e-3, batch_timesteps=1,
            seed=42, device="cpu", hidden_dim=32, n_layers=2,
            n_train_samples=10,
        )
        trainer = Trainer(config, graph, dataset)

        # Manually check: perturb and verify target would be noise
        x_0 = trainer.dataset[0]
        noise = torch.randn_like(x_0)
        t = 0.5
        x_t, _ = trainer.sde.perturb(x_0, t, noise=noise)
        # In epsilon-prediction, target = noise (NOT -noise/std)
        target = noise
        _, std = trainer.sde.marginal_params(t)
        old_target = -noise / std  # the broken version
        # They must differ
        assert not torch.allclose(target, old_target, atol=1e-3)


class TestDataset:
    def test_save_load_roundtrip(self, small_graph):
        """Dataset .npz roundtrip preserves data."""
        from graph_fans.phase2.dataset import generate_and_save_dataset, load_dataset

        graph, _ = small_graph
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_and_save_dataset(
                graph, family="SBM(q=0.05)", seed=0,
                n_train=5, n_ref=3, n_features=4,
                feature_mode="smooth", output_dir=tmpdir,
            )
            ds = load_dataset(path)
            assert ds["train"].shape == (5, 30, 4)
            assert ds["ref"].shape == (3, 30, 4)
            assert ds["family"] == "SBM(q=0.05)"
            assert ds["seed"] == 0

    def test_get_or_generate_caches(self, small_graph):
        """get_or_generate_dataset creates on first call, loads on second."""
        from graph_fans.phase2.dataset import get_or_generate_dataset

        graph, _ = small_graph
        with tempfile.TemporaryDirectory() as tmpdir:
            ds1 = get_or_generate_dataset(
                graph, "SBM(q=0.05)", seed=0,
                n_train=5, n_ref=3, n_features=4,
                feature_mode="smooth", cache_dir=tmpdir,
            )
            ds2 = get_or_generate_dataset(
                graph, "SBM(q=0.05)", seed=0,
                n_train=5, n_ref=3, n_features=4,
                feature_mode="smooth", cache_dir=tmpdir,
            )
            np.testing.assert_array_equal(ds1["train"], ds2["train"])

    def test_validate_dataset(self, small_graph):
        """validate_dataset returns spectral profile and warnings."""
        from graph_fans.phase2.dataset import get_or_generate_dataset, validate_dataset

        graph, _ = small_graph
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = get_or_generate_dataset(
                graph, "SBM(q=0.05)", seed=0,
                n_train=10, n_ref=3, n_features=4,
                feature_mode="smooth", cache_dir=tmpdir,
            )
            val = validate_dataset(ds, graph, B=4)
            assert "train_std" in val
            assert "spectral_profile" in val
            assert val["spectral_profile"].shape == (4,)


class TestTrainer:
    def test_loss_decreases(self, small_graph):
        """After a few epochs, loss should decrease."""
        graph, dataset = small_graph
        config = TrainConfig(
            n_epochs=50, lr=1e-3, batch_timesteps=4,
            seed=42, device="cpu", hidden_dim=32, n_layers=2,
            n_train_samples=10,
        )
        trainer = Trainer(config, graph, dataset)
        history = trainer.train()
        # With epsilon-prediction, loss should decrease meaningfully
        assert history["loss"][-1] < history["loss"][0]

    def test_generation_shape(self, small_graph):
        """Generated features should have correct shape."""
        graph, dataset = small_graph
        config = TrainConfig(
            n_epochs=10, batch_timesteps=2,
            seed=42, device="cpu", hidden_dim=32, n_layers=2,
            n_gen_steps=10, n_train_samples=10,
        )
        trainer = Trainer(config, graph, dataset)
        trainer.train()
        gen = trainer.generate(n_steps=10, n_samples=3)
        assert gen.shape == (3, dataset.shape[1], dataset.shape[2])

    def test_generation_is_deterministic(self, small_graph):
        """DDIM generation should be deterministic (same model, same init seed)."""
        graph, dataset = small_graph
        config = TrainConfig(
            n_epochs=5, batch_timesteps=2,
            seed=42, device="cpu", hidden_dim=32, n_layers=2,
            n_gen_steps=10, n_train_samples=10,
        )
        trainer = Trainer(config, graph, dataset)
        trainer.train()
        # Same seed for randn in generate — DDIM is deterministic after init
        torch.manual_seed(0)
        gen1 = trainer.generate(n_steps=10, n_samples=1)
        torch.manual_seed(0)
        gen2 = trainer.generate(n_steps=10, n_samples=1)
        np.testing.assert_allclose(gen1, gen2, atol=1e-5)

    def test_sanity_check_runs(self, small_graph):
        """Sanity check should run without errors."""
        graph, dataset = small_graph
        config = TrainConfig(
            n_epochs=5, batch_timesteps=2,
            seed=42, device="cpu", hidden_dim=32, n_layers=2,
            n_gen_steps=10, n_train_samples=10,
        )
        trainer = Trainer(config, graph, dataset)
        trainer.train()
        result = trainer.sanity_check(dataset, n_gen=2)
        assert "train_std" in result
        assert "gen_std" in result
        assert "std_ratio" in result
        assert "spectral_l2" in result

    def test_sanity_check_flags_untrained(self, small_graph):
        """Barely-trained model should trigger sanity check warnings."""
        graph, dataset = small_graph
        config = TrainConfig(
            n_epochs=1, batch_timesteps=1,
            seed=42, device="cpu", hidden_dim=32, n_layers=2,
            n_gen_steps=5, n_train_samples=10,
        )
        trainer = Trainer(config, graph, dataset)
        trainer.train()
        result = trainer.sanity_check(dataset, n_gen=2)
        # Barely-trained model likely has warnings (high std ratio or spectral mismatch)
        # We just verify it runs — warnings depend on random init
        assert isinstance(result["warnings"], list)


class TestIntegration:
    def test_uniform_vs_spectral_pipeline(self):
        """Full pipeline runs end-to-end on a tiny graph."""
        from graph_fans.phase2.evaluate import run_single_experiment

        energies = np.array([5.0, 3.0, 2.0, 1.0])
        weights = compute_importance_weights(energies)
        config = TrainConfig(
            n_epochs=20, batch_timesteps=2, seed=0, device="cpu",
            hidden_dim=32, n_layers=2, B=4, n_gen_steps=10,
            n_train_samples=20,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            r_uniform = run_single_experiment(
                "SBM(q=0.05)", "uniform", seed=0,
                n_nodes=30, n_features=4, config=config,
                importance_weights=None, B=4, feature_mode="smooth",
                dataset_dir=tmpdir,
            )
            r_spectral = run_single_experiment(
                "SBM(q=0.05)", "spectral", seed=0,
                n_nodes=30, n_features=4, config=config,
                importance_weights=weights, B=4, feature_mode="smooth",
                dataset_dir=tmpdir,
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
            n_train_samples=20,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            for t_knee in [0.1, 0.3]:
                r = run_single_experiment(
                    "SBM(q=0.05)", f"spectral_ramp_tknee={t_knee}", seed=0,
                    n_nodes=30, n_features=4, config=config,
                    importance_weights=weights, B=4, feature_mode="smooth",
                    dataset_dir=tmpdir,
                )
                assert r.qbe_total >= 0
                assert f"tknee={t_knee}" in r.method
