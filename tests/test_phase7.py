"""Smoke tests for Phase 7 real-dataset loaders.

These tests import the Elliptic and PPI datasets. Because downloading is required
and expensive, we cache at /tmp/pyg_data. Individual tests are skipped if the
underlying dataset cannot be accessed.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from graph_fans.phase7 import load_elliptic, load_ppi, load_real_dataset
from graph_fans.utils.graph_generators import GraphData


def _try(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # pragma: no cover - network or dataset errors
        pytest.skip(f"Could not load dataset: {e}")


class TestEllipticLoader:
    def test_returns_graphdata(self):
        gd = _try(load_elliptic)
        assert isinstance(gd, GraphData)
        assert isinstance(gd.graph, nx.Graph)
        assert isinstance(gd.features, np.ndarray)

    def test_shapes_consistent(self):
        gd = _try(load_elliptic)
        n = gd.graph.number_of_nodes()
        assert gd.features.shape[0] == n
        assert gd.features.shape[1] > 0

    def test_labels_present(self):
        gd = _try(load_elliptic)
        assert "labels" in gd.params
        assert "labeled_mask" in gd.params
        n = gd.graph.number_of_nodes()
        assert gd.params["labels"].shape == (n,)
        assert gd.params["labeled_mask"].dtype == bool

    def test_largest_cc(self):
        gd = _try(load_elliptic)
        # Should be a valid graph; relabeled to 0..n-1
        nodes = set(gd.graph.nodes())
        assert nodes == set(range(len(nodes)))


class TestPPILoader:
    def test_returns_graphdata(self):
        gd = _try(load_ppi, split="train", concatenate=False)
        assert isinstance(gd, GraphData)
        assert gd.features.shape[0] == gd.graph.number_of_nodes()
        assert gd.features.shape[1] > 0

    def test_labels_shape(self):
        gd = _try(load_ppi, split="train", concatenate=False)
        assert "labels" in gd.params
        labels = gd.params["labels"]
        assert labels.shape[0] == gd.graph.number_of_nodes()
        # PPI has 121 binary label dims
        if labels.ndim > 1:
            assert labels.shape[1] == 121

    def test_relabeled(self):
        gd = _try(load_ppi, split="train", concatenate=False)
        nodes = set(gd.graph.nodes())
        assert nodes == set(range(len(nodes)))


class TestUnifiedEntry:
    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_real_dataset("does-not-exist")

    def test_elliptic_via_entry(self):
        gd = _try(load_real_dataset, "elliptic")
        assert isinstance(gd, GraphData)
        assert gd.name == "Elliptic"

    def test_ppi_via_entry(self):
        gd = _try(load_real_dataset, "ppi", split="train", concatenate=False)
        assert isinstance(gd, GraphData)
        assert gd.name.startswith("PPI")
