"""Main script for Phase 1a: Basis-Free Spectral Metrics Validation.

Usage:
    uv run python -m graph_fans.phase1a.run_validation [--output-dir results/phase1a] [--n-nodes 200] [--bands 8]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import networkx as nx
import numpy as np

from graph_fans.utils.graph_generators import generate_sbm, generate_ba
from .metric_validator import MetricValidator
from .visualize import (
    plot_metric_comparison,
    plot_spectral_density_comparison,
    plot_hks_comparison,
    plot_information_gap,
    plot_g1_summary,
)
from graph_fans.phase0.spectral_profiler import compute_laplacian_spectrum

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _generate_er_baseline(graph_ref: nx.Graph, seed: int = 42) -> nx.Graph:
    """Generate Erdős-Rényi graph matching density of reference."""
    n = graph_ref.number_of_nodes()
    m = graph_ref.number_of_edges()
    p = 2 * m / (n * (n - 1)) if n > 1 else 0
    return nx.erdos_renyi_graph(n, p, seed=seed)


def _generate_config_baseline(graph_ref: nx.Graph, seed: int = 42) -> nx.Graph:
    """Generate configuration model matching degree sequence of reference."""
    deg_seq = [d for _, d in graph_ref.degree()]
    # Configuration model needs even sum of degrees
    if sum(deg_seq) % 2 != 0:
        deg_seq[0] += 1
    try:
        g = nx.configuration_model(deg_seq, seed=seed)
        g = nx.Graph(g)  # Remove multi-edges
        g.remove_edges_from(nx.selfloop_edges(g))
        return g
    except Exception:
        # Fallback to ER
        return _generate_er_baseline(graph_ref, seed)


def _generate_sbm_baseline(
    graph_ref: nx.Graph, n_communities: int = 4, seed: int = 42
) -> nx.Graph:
    """Generate SBM with approximately matched parameters."""
    n = graph_ref.number_of_nodes()
    m = graph_ref.number_of_edges()
    avg_degree = 2 * m / n if n > 0 else 0

    sizes = [n // n_communities] * n_communities
    sizes[-1] = n - sum(sizes[:-1])

    # Estimate p_intra and p_inter from average degree
    # Rough heuristic
    p_intra = min(avg_degree / (n // n_communities) * 1.5, 0.9)
    p_inter = max(avg_degree / n * 0.3, 0.001)

    p_matrix = np.full((n_communities, n_communities), p_inter)
    np.fill_diagonal(p_matrix, p_intra)

    return nx.stochastic_block_model(sizes, p_matrix.tolist(), seed=seed)


def _smooth_random_features(graph: nx.Graph, n_features: int = 16, seed: int = 42) -> np.ndarray:
    """Generate random features (not smooth) for baseline graphs."""
    rng = np.random.RandomState(seed)
    return rng.randn(graph.number_of_nodes(), n_features)


def run_validation(
    output_dir: str = "results/phase1a",
    n_nodes: int = 200,
    B: int = 8,
) -> dict:
    """Run Phase 1a metric validation."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate reference graphs
    ref_graphs_data = [
        generate_sbm(n_nodes=n_nodes, p_inter=0.01, seed=42),
        generate_sbm(n_nodes=n_nodes, p_inter=0.1, seed=42),
        generate_ba(n_nodes=n_nodes, m=3, seed=42),
        generate_ba(n_nodes=n_nodes, m=8, seed=42),
    ]

    # For each reference, generate 3 baselines of varying quality
    baseline_generators = [
        ("ER (matched density)", _generate_er_baseline),
        ("Config (matched degree)", _generate_config_baseline),
        ("SBM (approx params)", _generate_sbm_baseline),
    ]

    all_ref = []
    all_gen = []

    for gd in ref_graphs_data:
        for bl_name, bl_fn in baseline_generators:
            ref_graph = gd.graph
            gen_graph = bl_fn(ref_graph, seed=42)
            gen_features = _smooth_random_features(gen_graph, n_features=gd.features.shape[1])

            all_ref.append((ref_graph, gd.features, gd.name))
            all_gen.append((gen_graph, gen_features, bl_name))

    # Run validation
    validator = MetricValidator(all_ref, all_gen, B=B)
    validator.evaluate_all()

    # Summary
    df = validator.summary_table()
    logger.info(f"\n{df.to_string()}")
    df.to_csv(output_path / "metrics_summary.csv", index=False)

    # G1 decision
    g1 = validator.g1_decision()
    logger.info(f"\nG1 Decision: {g1['decision']}")
    logger.info(f"  Criterion: {g1['criterion']}")
    logger.info(f"  Any metric exceeds 10%: {g1['any_metric_exceeds_10pct']}")

    with open(output_path / "g1_decision.json", "w") as f:
        # Convert numpy types for JSON serialization
        g1_serializable = {}
        for k, v in g1.items():
            if isinstance(v, (np.floating, np.integer)):
                g1_serializable[k] = float(v)
            elif isinstance(v, dict):
                g1_serializable[k] = {
                    kk: {kkk: float(vvv) if isinstance(vvv, (np.floating, np.integer)) else vvv
                         for kkk, vvv in vv.items()} if isinstance(vv, dict) else vv
                    for kk, vv in v.items()
                }
            else:
                g1_serializable[k] = v
        json.dump(g1_serializable, f, indent=2)

    # Plots
    logger.info("\nGenerating plots...")
    plot_metric_comparison(df, save_path=output_path / "metric_comparison")
    plot_information_gap(df, save_path=output_path / "information_gap")
    plot_g1_summary(df, g1, save_path=output_path / "g1_summary")

    # Spectral density comparison plots for a few representative pairs
    for i, (gd, (bl_name, bl_fn)) in enumerate(
        zip(ref_graphs_data[:2], baseline_generators[:2])
    ):
        ref_graph = gd.graph
        gen_graph = bl_fn(ref_graph, seed=42)

        ev_ref, _ = compute_laplacian_spectrum(ref_graph)
        ev_gen, _ = compute_laplacian_spectrum(gen_graph)

        plot_spectral_density_comparison(
            ev_ref, ev_gen,
            title=f"{gd.name} vs {bl_name}",
            save_path=output_path / f"spectral_density_{i}",
        )
        plot_hks_comparison(
            ev_ref, ev_gen,
            title=f"{gd.name} vs {bl_name}",
            save_path=output_path / f"hks_{i}",
        )

    logger.info(f"\nResults saved to {output_path}/")
    return g1


def main():
    parser = argparse.ArgumentParser(description="Phase 1a: Spectral Metrics Validation")
    parser.add_argument("--output-dir", default="results/phase1a", help="Output directory")
    parser.add_argument("--n-nodes", type=int, default=200, help="Number of nodes")
    parser.add_argument("--bands", type=int, default=8, help="Number of spectral bands")
    args = parser.parse_args()

    run_validation(output_dir=args.output_dir, n_nodes=args.n_nodes, B=args.bands)


if __name__ == "__main__":
    main()
