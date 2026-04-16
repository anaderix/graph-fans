"""BFS subgraph extraction for real citation network experiments.

Samples connected induced subgraphs via BFS from random seed nodes,
returning relabeled graphs (nodes 0..n-1) suitable for eigendecomposition
and Trainer consumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass
class SubgraphSample:
    """A sampled subgraph with its original node mapping.

    Attributes:
        graph: Relabeled graph with nodes 0..n-1.
        node_indices: Original node IDs, shape [n]. node_indices[i] is the
            original ID of relabeled node i.
    """

    graph: nx.Graph
    node_indices: np.ndarray


def sample_bfs_subgraph(
    graph: nx.Graph, n_nodes: int, seed: int = 0
) -> SubgraphSample:
    """Sample a connected induced subgraph via BFS from a random seed node.

    If BFS doesn't reach n_nodes (disconnected region), take largest connected
    component of the BFS tree. If largest component < 0.8 * n_nodes, retry with
    a different seed node (up to 10 attempts).

    Args:
        graph: Source graph.
        n_nodes: Desired number of nodes in the subgraph.
        seed: Random seed for reproducibility.

    Returns:
        SubgraphSample with relabeled graph (nodes 0..n-1) and original node IDs.

    Raises:
        ValueError: If no valid subgraph could be sampled after 10 attempts.
    """
    rng = np.random.RandomState(seed)
    all_nodes = list(graph.nodes())

    if n_nodes > len(all_nodes):
        raise ValueError(
            f"Requested n_nodes={n_nodes} exceeds graph size={len(all_nodes)}"
        )

    best_sample: SubgraphSample | None = None
    best_size = 0

    for attempt in range(10):
        start_node = rng.choice(all_nodes)

        # BFS traversal collecting up to n_nodes
        visited: list[int] = []
        queue = [start_node]
        seen = {start_node}

        while queue and len(visited) < n_nodes:
            node = queue.pop(0)
            visited.append(node)
            for neighbor in graph.neighbors(node):
                if neighbor not in seen and len(visited) + len(queue) < n_nodes * 2:
                    seen.add(neighbor)
                    queue.append(neighbor)

        # Take exactly n_nodes if we have enough, otherwise take what we got
        visited = visited[:n_nodes]

        # Extract induced subgraph
        subgraph = graph.subgraph(visited).copy()

        # Get largest connected component
        components = list(nx.connected_components(subgraph))
        largest_cc = max(components, key=len)
        cc_size = len(largest_cc)

        if cc_size >= n_nodes:
            # Take exactly n_nodes from the largest component
            selected = sorted(largest_cc)[:n_nodes]
        elif cc_size >= 0.8 * n_nodes:
            # Accept the largest component even if slightly smaller
            selected = sorted(largest_cc)
        else:
            # Too small, track best and retry
            if cc_size > best_size:
                best_size = cc_size
                selected = sorted(largest_cc)
                node_indices = np.array(selected)
                sub = graph.subgraph(selected).copy()
                mapping = {old: new for new, old in enumerate(selected)}
                relabeled = nx.relabel_nodes(sub, mapping)
                best_sample = SubgraphSample(
                    graph=relabeled, node_indices=node_indices
                )
            continue

        # Build the relabeled subgraph
        node_indices = np.array(selected)
        sub = graph.subgraph(selected).copy()
        mapping = {old: new for new, old in enumerate(selected)}
        relabeled = nx.relabel_nodes(sub, mapping)

        return SubgraphSample(graph=relabeled, node_indices=node_indices)

    # If no attempt succeeded with >= 0.8 * n_nodes, return the best we found
    if best_sample is not None:
        return best_sample

    raise ValueError(
        f"Could not sample a connected subgraph of size >= {int(0.8 * n_nodes)} "
        f"from graph with {len(all_nodes)} nodes after 10 attempts."
    )


def sample_multiple_subgraphs(
    graph: nx.Graph, n_nodes: int, k: int, base_seed: int = 0
) -> list[SubgraphSample]:
    """Sample k distinct BFS subgraphs.

    Each subgraph uses a different seed derived from base_seed to ensure
    diversity. Subgraphs may share nodes but start from different BFS origins.

    Args:
        graph: Source graph.
        n_nodes: Desired number of nodes per subgraph.
        k: Number of subgraphs to sample.
        base_seed: Base random seed.

    Returns:
        List of k SubgraphSample instances.
    """
    samples = []
    for i in range(k):
        sample = sample_bfs_subgraph(graph, n_nodes, seed=base_seed + i)
        samples.append(sample)
    return samples
