"""Tests for Phase 6: Real Citation Network Validation.

Phase 6a tests: subgraph sampling, feature reduction, Gaussian augmentation,
and a full pipeline smoke test with Cora data.
"""

import networkx as nx
import numpy as np
import pytest

from graph_fans.phase6.subgraph_sampler import (
    SubgraphSample,
    sample_bfs_subgraph,
    sample_multiple_subgraphs,
)
from graph_fans.phase6.feature_reducer import fit_feature_reducer, reduce_features
from graph_fans.phase6.augmentation import (
    gaussian_augmentation,
    spectral_augmentation,
    dropout_augmentation,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def large_graph():
    """A medium-sized connected graph for subgraph sampling tests."""
    # Use an SBM with strong intra-community edges to ensure connectivity
    graph = nx.stochastic_block_model(
        [50, 50, 50, 50],
        [[0.3, 0.02, 0.02, 0.02],
         [0.02, 0.3, 0.02, 0.02],
         [0.02, 0.02, 0.3, 0.02],
         [0.02, 0.02, 0.02, 0.3]],
        seed=42,
    )
    # Remove block attributes
    for node in graph.nodes():
        if "block" in graph.nodes[node]:
            del graph.nodes[node]["block"]
    return graph


@pytest.fixture
def sparse_features():
    """Simulated sparse binary features like Cora."""
    rng = np.random.RandomState(42)
    # 200 nodes, 100 features, ~5% nonzero (sparse binary)
    features = (rng.rand(200, 100) < 0.05).astype(np.float64)
    return features


@pytest.fixture
def continuous_features():
    """Simulated continuous features for PCA testing."""
    rng = np.random.RandomState(42)
    # Features with some structure (not pure noise)
    base = rng.randn(200, 5)
    noise = rng.randn(200, 100) * 0.1
    features = base @ rng.randn(5, 100) + noise
    return features


# ============================================================================
# Subgraph sampling tests
# ============================================================================


class TestBFSSubgraph:
    def test_bfs_subgraph_size(self, large_graph):
        """Sampled subgraph has correct number of nodes."""
        sample = sample_bfs_subgraph(large_graph, n_nodes=30, seed=0)
        assert sample.graph.number_of_nodes() == 30

    def test_bfs_subgraph_connected(self, large_graph):
        """Sampled subgraph is connected."""
        sample = sample_bfs_subgraph(large_graph, n_nodes=30, seed=0)
        assert nx.is_connected(sample.graph)

    def test_bfs_subgraph_relabeled(self, large_graph):
        """Nodes are 0..n-1 after relabeling."""
        n = 30
        sample = sample_bfs_subgraph(large_graph, n_nodes=n, seed=0)
        assert set(sample.graph.nodes()) == set(range(n))

    def test_bfs_subgraph_node_indices(self, large_graph):
        """node_indices has correct shape and maps to valid original nodes."""
        n = 30
        sample = sample_bfs_subgraph(large_graph, n_nodes=n, seed=0)
        assert sample.node_indices.shape == (sample.graph.number_of_nodes(),)
        for idx in sample.node_indices:
            assert idx in large_graph.nodes()

    def test_bfs_subgraph_preserves_edges(self, large_graph):
        """Subgraph edges exist in the original graph."""
        sample = sample_bfs_subgraph(large_graph, n_nodes=30, seed=0)
        for u, v in sample.graph.edges():
            orig_u = sample.node_indices[u]
            orig_v = sample.node_indices[v]
            assert large_graph.has_edge(orig_u, orig_v)

    def test_bfs_subgraph_reproducible(self, large_graph):
        """Same seed produces same subgraph."""
        s1 = sample_bfs_subgraph(large_graph, n_nodes=30, seed=42)
        s2 = sample_bfs_subgraph(large_graph, n_nodes=30, seed=42)
        assert np.array_equal(s1.node_indices, s2.node_indices)
        assert set(s1.graph.edges()) == set(s2.graph.edges())

    def test_bfs_subgraph_different_seeds(self, large_graph):
        """Different seeds produce different subgraphs."""
        s1 = sample_bfs_subgraph(large_graph, n_nodes=30, seed=0)
        s2 = sample_bfs_subgraph(large_graph, n_nodes=30, seed=1)
        # Very unlikely to be identical with different seeds
        assert not np.array_equal(s1.node_indices, s2.node_indices)

    def test_bfs_subgraph_too_large(self, large_graph):
        """Raises ValueError if n_nodes exceeds graph size."""
        with pytest.raises(ValueError, match="exceeds graph size"):
            sample_bfs_subgraph(large_graph, n_nodes=300, seed=0)


class TestMultipleSubgraphs:
    def test_correct_count(self, large_graph):
        """Returns the requested number of subgraphs."""
        samples = sample_multiple_subgraphs(large_graph, n_nodes=30, k=5, base_seed=0)
        assert len(samples) == 5

    def test_all_correct_type(self, large_graph):
        """All samples are SubgraphSample instances."""
        samples = sample_multiple_subgraphs(large_graph, n_nodes=30, k=3, base_seed=0)
        for s in samples:
            assert isinstance(s, SubgraphSample)
            assert isinstance(s.graph, nx.Graph)
            assert isinstance(s.node_indices, np.ndarray)

    def test_diversity(self, large_graph):
        """Multiple subgraphs are not identical."""
        samples = sample_multiple_subgraphs(large_graph, n_nodes=30, k=5, base_seed=0)
        # Check that at least some pairs differ
        all_same = all(
            np.array_equal(samples[0].node_indices, s.node_indices)
            for s in samples[1:]
        )
        assert not all_same


# ============================================================================
# Feature reducer tests
# ============================================================================


class TestFeatureReducer:
    def test_truncated_svd_shape(self, sparse_features):
        """TruncatedSVD output has correct dimensions."""
        reducer = fit_feature_reducer(sparse_features, n_components=16, method="truncated_svd")
        reduced = reduce_features(sparse_features, reducer)
        assert reduced.shape == (200, 16)

    def test_pca_shape(self, continuous_features):
        """PCA output has correct dimensions."""
        reducer = fit_feature_reducer(continuous_features, n_components=8, method="pca")
        reduced = reduce_features(continuous_features, reducer)
        assert reduced.shape == (200, 8)

    def test_variance_preserved(self, continuous_features):
        """TruncatedSVD preserves most variance for structured features."""
        reducer = fit_feature_reducer(
            continuous_features, n_components=5, method="truncated_svd"
        )
        # 5 components should capture most variance in data with 5 latent dims
        variance_explained = reducer.explained_variance_ratio_.sum()
        assert variance_explained > 0.5  # at least 50%

    def test_subset_reduction(self, sparse_features):
        """Reducer fit on full data works on a subset."""
        reducer = fit_feature_reducer(sparse_features, n_components=10, method="truncated_svd")
        subset = sparse_features[:50]
        reduced = reduce_features(subset, reducer)
        assert reduced.shape == (50, 10)

    def test_reproducible(self, sparse_features):
        """Same seed produces identical reduction."""
        r1 = fit_feature_reducer(sparse_features, n_components=8, method="truncated_svd", seed=0)
        r2 = fit_feature_reducer(sparse_features, n_components=8, method="truncated_svd", seed=0)
        red1 = reduce_features(sparse_features, r1)
        red2 = reduce_features(sparse_features, r2)
        np.testing.assert_array_almost_equal(red1, red2)

    def test_invalid_method(self, sparse_features):
        """Unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            fit_feature_reducer(sparse_features, n_components=4, method="invalid")


# ============================================================================
# Gaussian augmentation tests
# ============================================================================


class TestGaussianAugmentation:
    def test_shape(self):
        """Output shape [N, n_nodes, n_features]."""
        features = np.random.randn(30, 4)
        aug = gaussian_augmentation(features, n_samples=50, seed=0)
        assert aug.shape == (50, 30, 4)

    def test_mean_close_to_original(self):
        """Mean of augmented samples is approximately the original features."""
        rng = np.random.RandomState(42)
        features = rng.randn(30, 4)
        aug = gaussian_augmentation(features, n_samples=500, sigma_range=(0.01, 0.1), seed=0)
        aug_mean = aug.mean(axis=0)
        # Mean should be close to original (noise averages out)
        np.testing.assert_allclose(aug_mean, features, atol=0.15)

    def test_augmented_differ_from_original(self):
        """Each augmented sample differs from the original."""
        features = np.random.randn(30, 4)
        aug = gaussian_augmentation(features, n_samples=10, seed=0)
        for i in range(10):
            assert not np.allclose(aug[i], features)

    def test_reproducible(self):
        """Same seed produces identical augmented data."""
        features = np.random.randn(30, 4)
        a1 = gaussian_augmentation(features, n_samples=20, seed=42)
        a2 = gaussian_augmentation(features, n_samples=20, seed=42)
        np.testing.assert_array_equal(a1, a2)

    def test_different_seeds(self):
        """Different seeds produce different augmented data."""
        features = np.random.randn(30, 4)
        a1 = gaussian_augmentation(features, n_samples=10, seed=0)
        a2 = gaussian_augmentation(features, n_samples=10, seed=1)
        assert not np.allclose(a1, a2)

    def test_sigma_range_affects_noise(self):
        """Wider sigma range produces more dispersed samples."""
        features = np.random.randn(30, 4)
        a_narrow = gaussian_augmentation(features, n_samples=200, sigma_range=(0.01, 0.02), seed=0)
        a_wide = gaussian_augmentation(features, n_samples=200, sigma_range=(0.3, 0.5), seed=0)
        # Wide sigma produces more dispersed samples
        std_narrow = a_narrow.std()
        std_wide = a_wide.std()
        assert std_wide > std_narrow

    def test_zero_std_features(self):
        """Handles constant features (zero std) gracefully."""
        features = np.ones((30, 4))
        aug = gaussian_augmentation(features, n_samples=10, seed=0)
        assert aug.shape == (10, 30, 4)
        # Should still produce noise (fallback std=1.0)
        assert not np.allclose(aug[0], features)


# ============================================================================
# Spectral augmentation tests (Phase 6b)
# ============================================================================


class TestSpectralAugmentation:
    @pytest.fixture
    def eigenvectors(self):
        """Eigenvectors from a small SBM graph."""
        graph = nx.stochastic_block_model(
            [15, 15], [[0.5, 0.05], [0.05, 0.5]], seed=42
        )
        for node in graph.nodes():
            if "block" in graph.nodes[node]:
                del graph.nodes[node]["block"]
        from graph_fans.phase0.spectral_profiler import compute_laplacian_spectrum
        _, evecs = compute_laplacian_spectrum(graph)
        return evecs

    def test_spectral_augmentation_shape(self, eigenvectors):
        """Output shape [N, n_nodes, n_features]."""
        n_nodes = eigenvectors.shape[0]
        features = np.random.RandomState(42).randn(n_nodes, 4)
        aug = spectral_augmentation(features, eigenvectors, n_samples=50, seed=0)
        assert aug.shape == (50, n_nodes, 4)

    def test_spectral_augmentation_preserves_spectral_profile(self, eigenvectors):
        """Augmented energy profile is approximately the same as original."""
        from graph_fans.phase0.spectral_profiler import partition_into_bands, compute_band_energy

        n_nodes = eigenvectors.shape[0]
        rng = np.random.RandomState(42)
        features = rng.randn(n_nodes, 8)

        evals = np.linspace(0, 2, n_nodes)
        _, bands = partition_into_bands(evals, B=4)

        orig_energy = compute_band_energy(features, eigenvectors, bands)
        orig_profile = orig_energy / orig_energy.sum()

        aug = spectral_augmentation(features, eigenvectors, n_samples=200, sigma=0.1, seed=0)
        aug_profiles = []
        for i in range(aug.shape[0]):
            e = compute_band_energy(aug[i], eigenvectors, bands)
            total = e.sum()
            if total > 0:
                aug_profiles.append(e / total)
        mean_aug_profile = np.mean(aug_profiles, axis=0)

        np.testing.assert_allclose(mean_aug_profile, orig_profile, atol=0.15)

    def test_spectral_augmentation_reproducible(self, eigenvectors):
        """Same seed produces identical results."""
        n_nodes = eigenvectors.shape[0]
        features = np.random.RandomState(42).randn(n_nodes, 4)
        a1 = spectral_augmentation(features, eigenvectors, n_samples=10, seed=0)
        a2 = spectral_augmentation(features, eigenvectors, n_samples=10, seed=0)
        np.testing.assert_array_equal(a1, a2)


# ============================================================================
# Dropout augmentation tests (Phase 6b)
# ============================================================================


class TestDropoutAugmentation:
    def test_dropout_augmentation_shape(self):
        """Output shape [N, n_nodes, n_features]."""
        features = np.random.RandomState(42).randn(30, 4)
        aug = dropout_augmentation(features, n_samples=50, seed=0)
        assert aug.shape == (50, 30, 4)

    def test_dropout_augmentation_sparsity(self):
        """Augmented has fewer nonzeros than original."""
        rng = np.random.RandomState(42)
        features = rng.randn(30, 8)
        aug = dropout_augmentation(features, n_samples=100, drop_rate=0.3, seed=0)
        orig_nonzero = np.count_nonzero(features)
        for i in range(aug.shape[0]):
            assert np.count_nonzero(aug[i]) <= orig_nonzero

    def test_dropout_augmentation_reproducible(self):
        """Same seed produces identical results."""
        features = np.random.RandomState(42).randn(30, 4)
        a1 = dropout_augmentation(features, n_samples=10, seed=0)
        a2 = dropout_augmentation(features, n_samples=10, seed=0)
        np.testing.assert_array_equal(a1, a2)

    def test_dropout_augmentation_rate(self):
        """Average fraction of zeros is approximately equal to drop_rate."""
        features = np.ones((100, 100))
        aug = dropout_augmentation(features, n_samples=200, drop_rate=0.3, seed=0)
        zero_frac = (aug == 0).mean()
        np.testing.assert_allclose(zero_frac, 0.3, atol=0.05)


# ============================================================================
# Full pipeline smoke test
# ============================================================================


class TestFullPipeline:
    @pytest.fixture
    def cora_data(self):
        """Load Cora data if available, skip otherwise."""
        try:
            from graph_fans.utils.graph_generators import load_citation_network
            return load_citation_network("Cora")
        except Exception:
            pytest.skip("Cora dataset not available (torch_geometric required)")

    def test_full_pipeline_smoke(self, cora_data):
        """Load Cora, sample 1 subgraph n=50, reduce to d=4, augment, train 5 epochs."""
        # Step 1: Fit reducer on full Cora features
        full_features = cora_data.features.astype(np.float64)
        reducer = fit_feature_reducer(full_features, n_components=4, method="truncated_svd")

        # Step 2: Sample one subgraph
        sample = sample_bfs_subgraph(cora_data.graph, n_nodes=50, seed=0)
        assert sample.graph.number_of_nodes() == 50
        assert nx.is_connected(sample.graph)

        # Step 3: Extract and reduce features
        sub_features = reduce_features(full_features[sample.node_indices], reducer)
        assert sub_features.shape == (50, 4)

        # Step 4: Augment
        train_data = gaussian_augmentation(sub_features, n_samples=20, seed=0)
        assert train_data.shape == (20, 50, 4)

        # Step 5: Train for 5 epochs (smoke test — just check no errors)
        from graph_fans.phase2.trainer import Trainer, TrainConfig

        cfg = TrainConfig(
            n_epochs=5,
            batch_timesteps=4,
            seed=0,
            device="cpu",
            hidden_dim=32,
            n_layers=2,
            conv_type="gcn",
            sde_type="cosine",
            use_ema=True,
            use_lr_scheduler=True,
            n_train_samples=20,
            noise_shaping="uniform",
        )
        trainer = Trainer(cfg, sample.graph, train_data)
        history = trainer.train()
        assert len(history["loss"]) == 5
        assert all(loss > 0 for loss in history["loss"])

        # Step 6: Generate a sample (smoke)
        gen = trainer.generate(n_samples=2)
        assert gen.shape == (2, 50, 4)


# ============================================================================
# Phase 6f: cross-protocol 2x2x2 shaping tests
# ============================================================================


class TestPhase6fCrossProtocol:
    def test_phase6f_script_imports(self):
        """Smoke test: the Phase 6f script imports without errors."""
        import importlib.util
        from pathlib import Path

        script_path = (
            Path(__file__).parent.parent / "scripts" / "test_phase6f_crossprotocol.py"
        )
        assert script_path.exists(), f"Script not found: {script_path}"

        spec = importlib.util.spec_from_file_location(
            "phase6f_script", script_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Required API
        assert hasattr(module, "run_seed")
        assert hasattr(module, "generate_independent_set")
        assert hasattr(module, "generate_specaug_set")
        assert hasattr(module, "compute_comparison")
        assert hasattr(module, "main")

    def test_crossprotocol_evaluation(self):
        """Verify each model is evaluated against BOTH reference sets.

        Runs a miniature seed (n_epochs=3, tiny model) and checks that the
        returned cells dict contains all 8 (A_uni, A_band, B_uni, B_band,
        C_uni, C_band, D_uni, D_band) keys with valid W1 numbers, and that
        the W1 values against indep-ref vs specaug-ref differ (meaning both
        refs were actually used).
        """
        import importlib.util
        from pathlib import Path

        script_path = (
            Path(__file__).parent.parent / "scripts" / "test_phase6f_crossprotocol.py"
        )
        spec = importlib.util.spec_from_file_location(
            "phase6f_script_eval", script_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Shrink to make this test fast
        module.N_NODES = 20
        module.N_FEATURES = 2
        module.N_TRAIN = 8
        module.N_REF = 6
        module.N_GEN = 5
        module.B = 4

        from graph_fans.phase0.spectral_profiler import (
            compute_laplacian_spectrum,
            partition_into_bands,
        )
        from graph_fans.utils.graph_generators import generate_sbm

        gd = generate_sbm(
            n_nodes=module.N_NODES, p_inter=0.05, seed=0, feature_mode="community",
        )
        graph = gd.graph
        evals, evecs = compute_laplacian_spectrum(graph)
        _, bands = partition_into_bands(evals, B=module.B)

        # Override TrainConfig-dependent kwargs inside run_seed by monkey-patching
        # via running with a very small n_epochs. Use CPU.
        entry = module.run_seed(
            graph, evecs, bands, seed=0, n_epochs=2, device="cpu",
        )

        assert "seed" in entry
        assert "cells" in entry
        expected_cells = {
            "A_uni", "A_band", "B_uni", "B_band",
            "C_uni", "C_band", "D_uni", "D_band",
        }
        assert set(entry["cells"].keys()) == expected_cells

        for name, cell in entry["cells"].items():
            assert "w1_total" in cell
            assert np.isfinite(cell["w1_total"])
            assert cell["w1_total"] >= 0

        # A_* and B_* share the same trained model (indep training) — their
        # W1 values should differ only via the reference protocol.
        a_uni = entry["cells"]["A_uni"]
        b_uni = entry["cells"]["B_uni"]
        assert a_uni["train_protocol"] == "indep"
        assert b_uni["train_protocol"] == "indep"
        assert a_uni["ref_protocol"] == "indep"
        assert b_uni["ref_protocol"] == "specaug"
        # Both references were used: W1 values should almost never coincide
        # by chance across 4 different (model, ref) pairings.
        w1s_indep_ref = [entry["cells"][k]["w1_total"] for k in ["A_uni", "A_band", "C_uni", "C_band"]]
        w1s_specaug_ref = [entry["cells"][k]["w1_total"] for k in ["B_uni", "B_band", "D_uni", "D_band"]]
        assert any(
            abs(w1s_indep_ref[i] - w1s_specaug_ref[i]) > 1e-6
            for i in range(len(w1s_indep_ref))
        ), "W1 values identical across refs — evaluation may not be using both refs"

    def test_compute_comparison_signature(self):
        """compute_comparison returns the expected summary fields."""
        import importlib.util
        from pathlib import Path

        script_path = (
            Path(__file__).parent.parent / "scripts" / "test_phase6f_crossprotocol.py"
        )
        spec = importlib.util.spec_from_file_location(
            "phase6f_script_cmp", script_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Two synthetic seeds with known W1 values
        per_seed = [
            {"cells": {"A_uni": {"w1_total": 100.0}, "A_band": {"w1_total": 90.0}}},
            {"cells": {"A_uni": {"w1_total": 110.0}, "A_band": {"w1_total": 95.0}}},
            {"cells": {"A_uni": {"w1_total": 120.0}, "A_band": {"w1_total": 100.0}}},
        ]
        comp = module.compute_comparison(per_seed, "A_uni", "A_band")
        for key in [
            "uniform_w1_mean", "band_w1_mean", "mean_delta_uni_minus_band",
            "improvement_pct", "t_stat", "p_val", "significant",
            "band_better_count", "n_seeds",
        ]:
            assert key in comp

        # With the seeded values (100,110,120) vs (90,95,100), band wins all
        # three pairs; delta mean should be +15; improvement ~13.6%.
        assert comp["band_better_count"] == 3
        assert comp["n_seeds"] == 3
        assert comp["mean_delta_uni_minus_band"] == pytest.approx(15.0, abs=0.01)
        assert comp["improvement_pct"] == pytest.approx(100 * 15.0 / 110.0, abs=0.01)
