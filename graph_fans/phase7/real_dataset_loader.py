"""Unified real-dataset loader for Phase 7.

Loads Elliptic Bitcoin and Stanford PPI datasets from torch_geometric and converts
them to the project-standard `GraphData` dataclass. Handles edge deduplication,
masks, and optional label/mask attachments via `params`.

The loaders return a single `GraphData` representing the largest connected component
(or for PPI, a concatenation of all 24 tissue PPIs — see `load_ppi`). Labels and
train/val/test masks (if available) are stored in `params`.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np

from graph_fans.utils.graph_generators import GraphData

logger = logging.getLogger(__name__)


def _edge_index_to_graph(edge_index: np.ndarray, n_nodes: int) -> nx.Graph:
    """Build an undirected NetworkX graph from a PyG edge_index, dedupe edges.

    Args:
        edge_index: shape [2, n_edges].
        n_nodes: number of nodes (to ensure isolated nodes are present).

    Returns:
        Undirected graph with nodes 0..n_nodes-1 and deduplicated edges.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    # Use a set to dedupe (u,v) and (v,u)
    edges = set()
    src, dst = edge_index[0], edge_index[1]
    for u, v in zip(src, dst):
        u = int(u)
        v = int(v)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        edges.add((a, b))
    graph.add_edges_from(edges)
    return graph


def _largest_connected_component(
    graph: nx.Graph, features: np.ndarray, extras: dict | None = None
) -> tuple[nx.Graph, np.ndarray, dict]:
    """Restrict to the largest connected component and relabel nodes.

    Args:
        graph: input graph.
        features: [n_nodes, d] node features.
        extras: optional dict of per-node arrays (labels, masks) to subset.

    Returns:
        Tuple of (relabeled_graph, subset_features, subset_extras).
    """
    components = list(nx.connected_components(graph))
    if not components:
        return graph, features, extras or {}
    largest = max(components, key=len)
    selected = sorted(largest)
    sub = graph.subgraph(selected).copy()
    mapping = {old: new for new, old in enumerate(selected)}
    relabeled = nx.relabel_nodes(sub, mapping)
    selected_arr = np.asarray(selected, dtype=np.int64)
    new_features = features[selected_arr]
    new_extras = {}
    if extras:
        for key, val in extras.items():
            if isinstance(val, np.ndarray) and val.shape[0] == features.shape[0]:
                new_extras[key] = val[selected_arr]
            else:
                new_extras[key] = val
    new_extras["original_node_indices"] = selected_arr
    return relabeled, new_features, new_extras


def _normalize_features(features: np.ndarray) -> np.ndarray:
    """Standardize features: zero-mean unit-variance per column, fallback to 1.0 std."""
    features = features.astype(np.float64)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-10] = 1.0
    return (features - mean) / std


def load_elliptic(root: str = "/tmp/pyg_data/elliptic", normalize: bool = True) -> GraphData:
    """Load Elliptic Bitcoin dataset via PyG.

    203k nodes, 166 continuous features (temporal aggregates), 3 classes
    (illicit=1, licit=0, unknown=2). Labels present only for labeled subset.

    Args:
        root: filesystem root for PyG dataset cache.
        normalize: if True, standardize features column-wise.

    Returns:
        GraphData with features, graph (largest CC), and extras in `params`:
        - `labels`: [n] int array; 2 = unknown
        - `labeled_mask`: [n] bool; True where labels in {0, 1}
        - `train_mask`, `test_mask`: temporal splits (from PyG)
        - `original_node_indices`: mapping back to original ids
    """
    try:
        from torch_geometric.datasets import EllipticBitcoinDataset
    except ImportError as e:
        raise ImportError("torch_geometric is required for Elliptic Bitcoin") from e

    ds = EllipticBitcoinDataset(root=root)
    data = ds[0]

    edge_index = data.edge_index.cpu().numpy()
    features = data.x.cpu().numpy()
    n = int(data.num_nodes)

    # Labels: PyG exposes `y` with 0=licit, 1=illicit, 2=unknown
    labels = data.y.cpu().numpy() if hasattr(data, "y") and data.y is not None else np.full(n, 2, dtype=np.int64)
    labeled_mask = labels != 2

    train_mask = (
        data.train_mask.cpu().numpy()
        if hasattr(data, "train_mask") and data.train_mask is not None
        else np.zeros(n, dtype=bool)
    )
    test_mask = (
        data.test_mask.cpu().numpy()
        if hasattr(data, "test_mask") and data.test_mask is not None
        else np.zeros(n, dtype=bool)
    )

    graph = _edge_index_to_graph(edge_index, n)
    extras = {
        "labels": labels,
        "labeled_mask": labeled_mask,
        "train_mask": train_mask,
        "test_mask": test_mask,
    }
    graph, features, extras = _largest_connected_component(graph, features, extras)

    if normalize:
        features = _normalize_features(features)
    else:
        features = features.astype(np.float64)

    logger.info(
        "Elliptic loaded: %d nodes, %d edges, %d features, %d labeled",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        features.shape[1],
        int(extras["labeled_mask"].sum()),
    )

    return GraphData(
        graph=graph,
        features=features,
        name="Elliptic",
        params={
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
            "n_features": features.shape[1],
            **extras,
        },
    )


def load_ppi(
    root: str = "/tmp/pyg_data/ppi",
    split: str = "train",
    concatenate: bool = True,
    normalize: bool = True,
) -> GraphData:
    """Load Stanford PPI dataset via PyG.

    PPI is 24 tissue-specific protein interaction graphs with 50 continuous
    positional gene features and 121 multi-label function annotations.

    Args:
        root: filesystem root.
        split: one of {"train", "val", "test"}. PyG splits PPI into 20/2/2 graphs.
        concatenate: if True, concatenate all graphs in the split into one
            disconnected graph (with block-diagonal structure). If False, return
            only the first graph in the split.
        normalize: if True, standardize features column-wise.

    Returns:
        GraphData with features and `labels` (shape [n, 121] float) in params.
    """
    try:
        from torch_geometric.datasets import PPI
    except ImportError as e:
        raise ImportError("torch_geometric is required for PPI") from e

    ds = PPI(root=root, split=split)

    if not concatenate:
        data = ds[0]
        edge_index = data.edge_index.cpu().numpy()
        features = data.x.cpu().numpy()
        labels = data.y.cpu().numpy()
        n = int(data.num_nodes)
        graph = _edge_index_to_graph(edge_index, n)
        extras = {"labels": labels}
        graph, features, extras = _largest_connected_component(graph, features, extras)
        if normalize:
            features = _normalize_features(features)
        else:
            features = features.astype(np.float64)
        return GraphData(
            graph=graph,
            features=features,
            name=f"PPI-{split}-single",
            params={
                "n_nodes": graph.number_of_nodes(),
                "n_edges": graph.number_of_edges(),
                "n_features": features.shape[1],
                "n_label_dims": int(labels.shape[1]) if labels.ndim > 1 else 1,
                **extras,
            },
        )

    # Concatenate: build a single block-diagonal graph
    all_edges: list[tuple[int, int]] = []
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    offset = 0
    graph_ids: list[np.ndarray] = []
    for gi, data in enumerate(ds):
        edge_index = data.edge_index.cpu().numpy()
        n = int(data.num_nodes)
        for u, v in zip(edge_index[0], edge_index[1]):
            a, b = int(u) + offset, int(v) + offset
            if a == b:
                continue
            all_edges.append((min(a, b), max(a, b)))
        all_features.append(data.x.cpu().numpy())
        all_labels.append(data.y.cpu().numpy())
        graph_ids.append(np.full(n, gi, dtype=np.int64))
        offset += n

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    graph_id_arr = np.concatenate(graph_ids, axis=0)
    n_total = offset

    graph = nx.Graph()
    graph.add_nodes_from(range(n_total))
    graph.add_edges_from(set(all_edges))

    # For subgraph sampling to work, we want a connected graph. PPI concat is
    # disconnected (one CC per tissue). We take the largest CC.
    extras = {"labels": labels, "graph_id": graph_id_arr}
    graph, features, extras = _largest_connected_component(graph, features, extras)

    if normalize:
        features = _normalize_features(features)
    else:
        features = features.astype(np.float64)

    logger.info(
        "PPI (%s, concat=%s) loaded: %d nodes, %d edges, %d features, %d label dims",
        split,
        concatenate,
        graph.number_of_nodes(),
        graph.number_of_edges(),
        features.shape[1],
        int(extras["labels"].shape[1]) if extras["labels"].ndim > 1 else 1,
    )

    return GraphData(
        graph=graph,
        features=features,
        name=f"PPI-{split}",
        params={
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
            "n_features": features.shape[1],
            "n_label_dims": int(extras["labels"].shape[1]) if extras["labels"].ndim > 1 else 1,
            **extras,
        },
    )


def load_real_dataset(name: str, **kwargs: Any) -> GraphData:
    """Unified entry point for Phase 7 datasets.

    Args:
        name: one of {"elliptic", "ppi", "cora", "citeseer"}. Case-insensitive.
        **kwargs: forwarded to the specific loader.

    Returns:
        GraphData.
    """
    key = name.strip().lower()
    if key == "elliptic":
        return load_elliptic(**kwargs)
    if key == "ppi":
        return load_ppi(**kwargs)
    if key in {"cora", "citeseer"}:
        from graph_fans.utils.graph_generators import load_citation_network

        return load_citation_network(name.capitalize())
    raise ValueError(f"Unknown dataset: {name!r}. Expected elliptic|ppi|cora|citeseer.")
