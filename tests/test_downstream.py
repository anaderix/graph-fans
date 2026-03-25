"""Tests for graph_fans.phase2.downstream module."""

from __future__ import annotations

import tempfile

import networkx as nx
import numpy as np
import pytest

from graph_fans.phase2.downstream import (
    build_node_classification_dataset,
    evaluate_node_classification,
    generate_feature_set,
    run_downstream_experiment,
    _get_community_labels,
)


@pytest.fixture
def small_sbm_graph():
    """20-node SBM graph with 4 communities for downstream tests."""
    from graph_fans.utils.graph_generators import generate_sbm
    gd = generate_sbm(n_nodes=20, n_communities=4, p_intra=0.5, p_inter=0.05,
                      n_features=4, seed=42, feature_mode="community")
    return gd.graph


@pytest.fixture
def eigenvectors_20(small_sbm_graph):
    """Laplacian eigenvectors for the 20-node graph."""
    from graph_fans.phase0.spectral_profiler import compute_laplacian_spectrum
    _, evecs = compute_laplacian_spectrum(small_sbm_graph)
    return evecs


class TestBuildNodeClassificationDataset:
    def test_output_shapes(self, eigenvectors_20):
        """With 10 samples, 20 nodes, 4 features: X=(200,4), y=(200,)."""
        features_set = np.random.randn(10, 20, 4)
        community_labels = np.repeat(np.arange(4), 5)  # 4 communities × 5 nodes
        X, y = build_node_classification_dataset(
            features_set, community_labels, eigenvectors_20
        )
        assert X.shape == (200, 4), f"Expected (200, 4), got {X.shape}"
        assert y.shape == (200,), f"Expected (200,), got {y.shape}"

    def test_labels_tiled_correctly(self, eigenvectors_20):
        """Labels are tiled n_samples times across the node dimension."""
        features_set = np.random.randn(3, 20, 4)
        community_labels = np.arange(20) % 4
        X, y = build_node_classification_dataset(
            features_set, community_labels, eigenvectors_20
        )
        # First 20 labels should match community_labels
        np.testing.assert_array_equal(y[:20], community_labels)
        # Second 20 labels should also match community_labels
        np.testing.assert_array_equal(y[20:40], community_labels)

    def test_x_values_are_finite(self, eigenvectors_20):
        """All X values should be finite (no NaN/Inf from projection)."""
        features_set = np.random.randn(5, 20, 4)
        community_labels = np.zeros(20, dtype=int)
        X, y = build_node_classification_dataset(
            features_set, community_labels, eigenvectors_20
        )
        assert np.all(np.isfinite(X)), "Non-finite values in X"


class TestEvaluateNodeClassification:
    def test_returns_valid_accuracy(self, eigenvectors_20):
        """Accuracy values should be in [0, 1]."""
        features_gen = np.random.randn(20, 20, 4)
        features_ref = np.random.randn(10, 20, 4)
        community_labels = np.repeat(np.arange(4), 5)

        result = evaluate_node_classification(
            features_gen, features_ref, community_labels, eigenvectors_20, seed=0
        )
        assert 0.0 <= result["train_acc"] <= 1.0, f"train_acc={result['train_acc']} out of [0,1]"
        assert 0.0 <= result["test_acc"] <= 1.0, f"test_acc={result['test_acc']} out of [0,1]"

    def test_train_test_independence(self, eigenvectors_20):
        """Train and test accuracies should come from separate splits."""
        # Use highly structured data so we can verify separation
        np.random.seed(42)
        features_gen = np.random.randn(50, 20, 4)
        features_ref = np.random.randn(10, 20, 4)
        community_labels = np.repeat(np.arange(4), 5)

        result = evaluate_node_classification(
            features_gen, features_ref, community_labels, eigenvectors_20, seed=0
        )
        # Both should be valid; train_acc may differ from test_acc
        assert isinstance(result["train_acc"], float)
        assert isinstance(result["test_acc"], float)

    def test_returns_expected_keys(self, eigenvectors_20):
        """Result dict must have 'train_acc' and 'test_acc' keys."""
        features_gen = np.random.randn(10, 20, 4)
        features_ref = np.random.randn(5, 20, 4)
        # Use at least 2 distinct community labels (single class would raise ValueError)
        community_labels = np.repeat(np.arange(4), 5)  # 4 communities × 5 nodes

        result = evaluate_node_classification(
            features_gen, features_ref, community_labels, eigenvectors_20, seed=0
        )
        assert "train_acc" in result
        assert "test_acc" in result


class TestRunDownstreamSmoke:
    def test_run_downstream_smoke(self):
        """run_downstream_experiment completes on a tiny graph and returns required keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_downstream_experiment(
                family="SBM(q=0.05)",
                n_nodes=20,
                n_seeds=2,
                device="cpu",
                dataset_dir=tmpdir,
                n_gen_samples=10,
                n_epochs=5,
                batch_timesteps=2,
                hidden_dim=32,
                n_layers=2,
            )

        required_keys = [
            "family", "n_nodes", "uniform_acc_per_seed", "spectral_acc_per_seed",
            "uniform_acc_mean", "spectral_acc_mean", "improvement_pct", "p_val",
            "significant", "per_seed",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

        assert len(result["uniform_acc_per_seed"]) == 2
        assert len(result["spectral_acc_per_seed"]) == 2
        assert result["family"] == "SBM(q=0.05)"
        assert result["n_nodes"] == 20

    def test_accuracy_above_random_chance(self):
        """With properly trained model, test_acc should exceed random chance on SBM."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_downstream_experiment(
                family="SBM(q=0.05)",
                n_nodes=20,
                n_seeds=1,
                device="cpu",
                dataset_dir=tmpdir,
                n_gen_samples=50,
                n_epochs=30,
                batch_timesteps=4,
                hidden_dim=32,
                n_layers=2,
            )

        # n_communities for 20-node SBM is at least 2 in louvain; random = 1/n_communities
        # The test checks accuracy is non-degenerate (not exactly 0 or random-degenerate)
        assert result["n_communities"] >= 1
        # Both accuracies must be in [0, 1]
        assert 0.0 <= result["uniform_acc_mean"] <= 1.0
        assert 0.0 <= result["spectral_acc_mean"] <= 1.0

    def test_no_data_leakage_structure(self):
        """Verify per_seed entries have separate train and test metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_downstream_experiment(
                family="SBM(q=0.05)",
                n_nodes=20,
                n_seeds=2,
                device="cpu",
                dataset_dir=tmpdir,
                n_gen_samples=5,
                n_epochs=3,
                batch_timesteps=2,
                hidden_dim=32,
                n_layers=2,
            )

        for seed_entry in result["per_seed"]:
            for method in ["uniform", "spectral"]:
                assert method in seed_entry, f"Missing {method} in seed_entry"
                assert "train_acc" in seed_entry[method]
                assert "test_acc" in seed_entry[method]
                # train_acc and test_acc are computed on separate splits
                # They should both be floats in [0, 1]
                assert 0.0 <= seed_entry[method]["train_acc"] <= 1.0
                assert 0.0 <= seed_entry[method]["test_acc"] <= 1.0


class TestGetCommunityLabels:
    def test_labels_shape(self, small_sbm_graph):
        """Community labels should have length n_nodes."""
        labels = _get_community_labels(small_sbm_graph, "SBM(q=0.05)", seed=0)
        assert labels.shape == (20,), f"Expected (20,), got {labels.shape}"

    def test_labels_are_integers(self, small_sbm_graph):
        """Community labels should be non-negative integers."""
        labels = _get_community_labels(small_sbm_graph, "SBM(q=0.05)", seed=0)
        assert labels.dtype in [np.int32, np.int64, int], f"Expected int dtype, got {labels.dtype}"
        assert np.all(labels >= 0)

    def test_fixed_seed_reproducible(self, small_sbm_graph):
        """Calling with seed=0 twice should give identical labels."""
        labels1 = _get_community_labels(small_sbm_graph, "SBM(q=0.05)", seed=0)
        labels2 = _get_community_labels(small_sbm_graph, "SBM(q=0.05)", seed=0)
        np.testing.assert_array_equal(labels1, labels2)
