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
from graph_fans.phase6.augmentation import gaussian_augmentation


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
