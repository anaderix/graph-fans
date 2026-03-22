"""Validation framework for Phase 1a: Basis-Free Spectral Metrics."""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd

from .spectral_metrics import SpectralMetrics, MetricResults

logger = logging.getLogger(__name__)


class MetricValidator:
    """Validate spectral metrics against standard metrics on graph pairs.

    Determines whether spectral metrics reveal fidelity gaps invisible
    to standard degree/clustering/orbit MMD metrics (G1 gate).
    """

    def __init__(
        self,
        reference_graphs: list[tuple[nx.Graph, np.ndarray, str]],
        generated_graphs: list[tuple[nx.Graph, np.ndarray, str]],
        B: int = 8,
    ):
        """
        Args:
            reference_graphs: List of (graph, features, name) tuples.
            generated_graphs: List of (graph, features, name) tuples.
                Must be same length as reference_graphs (paired).
            B: Number of spectral bands.
        """
        assert len(reference_graphs) == len(generated_graphs)
        self.ref_graphs = reference_graphs
        self.gen_graphs = generated_graphs
        self.metrics = SpectralMetrics(B=B)
        self.results: list[tuple[str, str, MetricResults]] = []

    def evaluate_all(self) -> list[tuple[str, str, MetricResults]]:
        """Compute all metrics for each reference-generated pair."""
        self.results = []
        for (g_ref, f_ref, name_ref), (g_gen, f_gen, name_gen) in zip(
            self.ref_graphs, self.gen_graphs
        ):
            logger.info(f"Evaluating: {name_ref} vs {name_gen}")
            result = self.metrics.compute_all(g_ref, f_ref, g_gen, f_gen)
            self.results.append((name_ref, name_gen, result))
            logger.info(
                f"  JSD={result.spectral_density_jsd:.4f}, "
                f"QBE={result.quantile_band_energy_dist:.4f}, "
                f"HKS={result.hks_distance:.4f} | "
                f"deg_MMD={result.degree_mmd:.4f}, "
                f"clust_MMD={result.clustering_mmd:.4f}, "
                f"tri_MMD={result.triangle_mmd:.4f}"
            )
        return self.results

    def summary_table(self) -> pd.DataFrame:
        """Return DataFrame with all metrics side by side."""
        if not self.results:
            self.evaluate_all()

        rows = []
        for ref_name, gen_name, r in self.results:
            rows.append(
                {
                    "Reference": ref_name,
                    "Generated": gen_name,
                    "Spectral JSD": r.spectral_density_jsd,
                    "QBE Distance": r.quantile_band_energy_dist,
                    "HKS Distance": r.hks_distance,
                    "Spectral Aggregate": r.spectral_aggregate(),
                    "Degree MMD": r.degree_mmd,
                    "Clustering MMD": r.clustering_mmd,
                    "Triangle MMD": r.triangle_mmd,
                    "Standard Aggregate": r.standard_aggregate(),
                }
            )
        return pd.DataFrame(rows)

    def compute_information_gap(self) -> dict:
        """Quantify additional information from spectral metrics beyond standard ones.

        For each pair, compute relative difference between spectral and standard
        metric rankings. Reports if any spectral metric shows ≥10% relative
        difference from aggregate MMD (G1 criterion).

        Returns:
            Dict with gap analysis results.
        """
        if not self.results:
            self.evaluate_all()

        df = self.summary_table()

        # Rank pairs by each metric (higher = worse fidelity)
        spectral_metrics = ["Spectral JSD", "QBE Distance", "HKS Distance"]
        standard_metrics = ["Degree MMD", "Clustering MMD", "Triangle MMD"]

        gaps = {}
        for sm in spectral_metrics:
            # Relative difference between spectral metric and standard aggregate
            for _, row in df.iterrows():
                std_agg = row["Standard Aggregate"]
                spec_val = row[sm]
                if std_agg > 1e-10:
                    rel_diff = abs(spec_val - std_agg) / std_agg
                else:
                    rel_diff = spec_val if spec_val > 1e-10 else 0.0

                key = f"{row['Reference']} vs {row['Generated']}"
                gaps.setdefault(key, {})[sm] = rel_diff

        # Check G1 criterion: any spectral metric shows ≥10% relative difference
        max_gaps = {}
        for pair, metric_gaps in gaps.items():
            max_gap = max(metric_gaps.values())
            max_metric = max(metric_gaps, key=metric_gaps.get)
            max_gaps[pair] = {"max_gap": max_gap, "metric": max_metric}

        any_passes = any(g["max_gap"] >= 0.10 for g in max_gaps.values())

        # Also check rank correlation between spectral and standard ordering
        if len(df) >= 3:
            from scipy.stats import spearmanr

            rank_corr, rank_p = spearmanr(
                df["Spectral Aggregate"], df["Standard Aggregate"]
            )
        else:
            rank_corr, rank_p = float("nan"), float("nan")

        return {
            "per_pair_gaps": gaps,
            "max_gaps": max_gaps,
            "any_metric_exceeds_10pct": any_passes,
            "rank_correlation": rank_corr,
            "rank_p_value": rank_p,
        }

    def g1_decision(self) -> dict:
        """Return Go/No-Go decision based on G1 criteria.

        G1 passes if at least 1 of 3 proposed metrics shows ≥10% relative
        difference from aggregate MMD on existing baselines.
        """
        gap_analysis = self.compute_information_gap()

        decision = "GO" if gap_analysis["any_metric_exceeds_10pct"] else "NO-GO"

        return {
            "decision": decision,
            "criterion": "At least 1 spectral metric shows ≥10% relative difference from aggregate MMD",
            "any_metric_exceeds_10pct": gap_analysis["any_metric_exceeds_10pct"],
            "max_gaps": gap_analysis["max_gaps"],
            "rank_correlation": gap_analysis["rank_correlation"],
            "rank_p_value": gap_analysis["rank_p_value"],
        }
