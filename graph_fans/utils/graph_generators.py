"""Graph generation utilities for Graph-FANS experiments."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
from scipy.linalg import expm

logger = logging.getLogger(__name__)


@dataclass
class GraphData:
    """Container for a graph with node features."""

    graph: nx.Graph
    features: np.ndarray  # shape [n_nodes, n_features]
    name: str
    params: dict[str, Any] = field(default_factory=dict)


def _smooth_features(
    graph: nx.Graph, n_features: int = 16, t: float = 1.0, seed: int = 42
) -> np.ndarray:
    """Generate smooth node features by low-pass filtering random signals on the graph.

    Creates random signals and applies exp(-t*L) to produce features
    with energy concentrated in low-frequency (low-eigenvalue) bands.
    """
    rng = np.random.RandomState(seed)
    n = graph.number_of_nodes()
    L = nx.normalized_laplacian_matrix(graph).toarray()
    raw = rng.randn(n, n_features)
    # Low-pass filter: exp(-t * L) applied to each feature column
    filt = expm(-t * L)
    features = filt @ raw
    # Normalize per feature to unit variance
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-10] = 1.0
    features = features / std
    return features


def generate_sbm(
    n_nodes: int = 200,
    n_communities: int = 4,
    p_intra: float = 0.3,
    p_inter: float = 0.01,
    n_features: int = 16,
    seed: int = 42,
) -> GraphData:
    """Generate a Stochastic Block Model graph with smooth node features.

    Args:
        n_nodes: Total number of nodes.
        n_communities: Number of communities.
        p_intra: Within-community edge probability.
        p_inter: Between-community edge probability (modularity parameter q).
        n_features: Number of node features.
        seed: Random seed.
    """
    sizes = [n_nodes // n_communities] * n_communities
    # Adjust last community to account for rounding
    sizes[-1] = n_nodes - sum(sizes[:-1])
    p_matrix = np.full((n_communities, n_communities), p_inter)
    np.fill_diagonal(p_matrix, p_intra)

    graph = nx.stochastic_block_model(sizes, p_matrix.tolist(), seed=seed)
    # Remove block attribute to keep it a simple graph
    for node in graph.nodes():
        if "block" in graph.nodes[node]:
            del graph.nodes[node]["block"]

    features = _smooth_features(graph, n_features=n_features, seed=seed)

    return GraphData(
        graph=graph,
        features=features,
        name=f"SBM(q={p_inter})",
        params={
            "n_nodes": n_nodes,
            "n_communities": n_communities,
            "p_intra": p_intra,
            "p_inter": p_inter,
        },
    )


def generate_ba(
    n_nodes: int = 200,
    m: int = 5,
    n_features: int = 16,
    seed: int = 42,
) -> GraphData:
    """Generate a Barabási-Albert preferential attachment graph with smooth node features."""
    graph = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
    features = _smooth_features(graph, n_features=n_features, seed=seed)

    return GraphData(
        graph=graph,
        features=features,
        name=f"BA(m={m})",
        params={"n_nodes": n_nodes, "m": m},
    )


def load_citation_network(
    name: str = "Cora", n_features: int | None = None
) -> GraphData:
    """Load a citation network from torch_geometric.

    Args:
        name: Dataset name ('Cora' or 'CiteSeer').
        n_features: If set, truncate features to this many dimensions.
    """
    try:
        from torch_geometric.datasets import Planetoid
        import torch
    except ImportError:
        raise ImportError(
            "torch_geometric is required for citation networks. "
            "Install with: pip install torch-geometric"
        )

    dataset = Planetoid(root="/tmp/pyg_data", name=name)
    data = dataset[0]

    # Convert to networkx
    edge_index = data.edge_index.numpy()
    graph = nx.Graph()
    graph.add_nodes_from(range(data.num_nodes))
    edges = list(zip(edge_index[0], edge_index[1]))
    graph.add_edges_from(edges)

    # Use actual node features
    features = data.x.numpy()
    if n_features is not None:
        features = features[:, :n_features]

    return GraphData(
        graph=graph,
        features=features,
        name=name,
        params={"n_nodes": data.num_nodes, "n_edges": data.num_edges},
    )


def get_all_graph_families(
    n_nodes: int = 200, n_features: int = 16, seed: int = 42
) -> list[GraphData]:
    """Generate all graph families for Phase 0 profiling."""
    graphs = []

    # SBM with varying modularity
    for q in [0.01, 0.05, 0.1, 0.2]:
        graphs.append(
            generate_sbm(
                n_nodes=n_nodes,
                p_intra=0.3,
                p_inter=q,
                n_features=n_features,
                seed=seed,
            )
        )

    # Barabási-Albert with varying attachment
    for m in [2, 5, 10]:
        graphs.append(
            generate_ba(n_nodes=n_nodes, m=m, n_features=n_features, seed=seed)
        )

    # Citation networks
    for name in ["Cora", "CiteSeer"]:
        try:
            graphs.append(load_citation_network(name, n_features=n_features))
        except Exception as e:
            logger.warning(f"Could not load {name}: {e}")

    return graphs
