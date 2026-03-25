"""Analyze family generalization results from test_shaping_w1.py output.

Loads results/diagnostics/family_generalization.json and produces:
  - Summary markdown table
  - Grouped bar chart (uniform vs spectral W1 per family)
  - Bimodality correlation scatter (improvement% vs bimodality index)

Usage:
    uv run python scripts/analyze_family_results.py \
        --input results/diagnostics/family_generalization.json \
        --output-dir results/diagnostics
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from graph_fans.phase0.spectral_profiler import (
    compute_laplacian_spectrum,
    partition_into_bands,
    compute_band_energy,
)
from graph_fans.phase2.evaluate import _get_graph
from graph_fans.utils.graph_generators import generate_sbm, generate_ba

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def load_results(path: str) -> pd.DataFrame:
    """Load family generalization JSON into a summary DataFrame.

    Args:
        path: Path to JSON file produced by test_shaping_w1.py.

    Returns:
        DataFrame with columns: family, n_nodes, uniform_w1_mean, uniform_w1_std,
        spectral_w1_mean, spectral_w1_std, improvement_pct, p_val, significant.
    """
    with open(path) as f:
        data = json.load(f)

    rows = []
    for entry in data:
        rows.append({
            "family": entry["family"],
            "n_nodes": entry.get("n_nodes", 50),
            "uniform_w1_mean": entry["uniform_w1_mean"],
            "uniform_w1_std": entry.get("uniform_w1_std", 0.0),
            "spectral_w1_mean": entry["spectral_w1_mean"],
            "spectral_w1_std": entry.get("spectral_w1_std", 0.0),
            "improvement_pct": entry["improvement_pct"],
            "p_val": entry["p_val"],
            "significant": entry["significant"],
        })
    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} family results from {path}")
    return df


def _compute_bimodality_index(family: str, n_nodes: int, seed: int = 0) -> float:
    """Compute bimodality index as max_band_energy / mean_band_energy.

    Uses a fresh graph at seed=0 with B=8 bands. Spectral profile derived
    from the graph topology only (Laplacian eigenvectors), not from features.

    Args:
        family: Graph family string.
        n_nodes: Number of nodes.
        seed: Random seed for graph topology (default 0).

    Returns:
        Bimodality index >= 1.0. Higher means more peaked/bimodal spectrum.
    """
    graph = _get_graph(family, n_nodes, seed=seed)
    evals, evecs = compute_laplacian_spectrum(graph)
    _, bands = partition_into_bands(evals, B=8)

    # Use eigenvector norms as a proxy for per-band energy without features
    band_energies = np.array([
        np.sum(evecs[:, idx] ** 2) / max(len(idx), 1)
        for idx in bands
    ])
    band_energies = band_energies / band_energies.sum()
    mean_e = band_energies.mean()
    return float(band_energies.max() / max(mean_e, 1e-10))


def plot_family_comparison(df: pd.DataFrame, save_path: str) -> None:
    """Grouped bar chart: uniform vs spectral W1 per family with error bars.

    Args:
        df: DataFrame from load_results().
        save_path: Path (without extension) to save the PNG.
    """
    families = df["family"].tolist()
    x = np.arange(len(families))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_u = ax.bar(
        x - width / 2,
        df["uniform_w1_mean"],
        width,
        yerr=df["uniform_w1_std"],
        label="Uniform",
        color="steelblue",
        alpha=0.85,
        capsize=4,
    )
    bars_s = ax.bar(
        x + width / 2,
        df["spectral_w1_mean"],
        width,
        yerr=df["spectral_w1_std"],
        label="Graph-FANS (spectral)",
        color="coral",
        alpha=0.85,
        capsize=4,
    )

    # Add significance stars above spectral bars
    for i, (_, row) in enumerate(df.iterrows()):
        if row["significant"]:
            y_top = row["spectral_w1_mean"] + row["spectral_w1_std"] + 1.0
            ax.text(i + width / 2, y_top, "*", ha="center", va="bottom",
                    fontsize=14, color="green", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=15, ha="right")
    ax.set_ylabel("Total W1 Distance (lower is better)")
    ax.set_title("Family Generalization: Uniform vs Spectral Noise")
    ax.legend()
    fig.tight_layout()

    out = Path(save_path)
    fig.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    md_path = str(out) + ".md"
    with open(md_path, "w") as f:
        f.write("# Family Comparison: W1 Distance\n\n")
        f.write("Grouped bar chart showing total W1 distance (lower is better) ")
        f.write("for uniform vs spectral noise per graph family. ")
        f.write("* = significant improvement (p<0.025, Bonferroni-adjusted).\n")
    logger.info(f"Saved {out}.png and {md_path}")
    plt.close(fig)


def plot_bimodality_correlation(
    df: pd.DataFrame,
    save_path: str,
) -> None:
    """Scatter: improvement% vs bimodality index per family.

    Args:
        df: DataFrame from load_results().
        save_path: Path (without extension) to save the PNG.
    """
    bimodality = []
    for _, row in df.iterrows():
        bi = _compute_bimodality_index(row["family"], int(row["n_nodes"]))
        bimodality.append(bi)
    df = df.copy()
    df["bimodality_index"] = bimodality

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["green" if s else "gray" for s in df["significant"]]
    ax.scatter(df["bimodality_index"], df["improvement_pct"],
               c=colors, s=120, edgecolors="white", zorder=3)

    for _, row in df.iterrows():
        ax.annotate(
            row["family"],
            (row["bimodality_index"], row["improvement_pct"]),
            xytext=(5, 5), textcoords="offset points", fontsize=9,
        )

    ax.axhline(y=0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("Bimodality Index (max_band_energy / mean_band_energy)")
    ax.set_ylabel("W1 Improvement (%)")
    ax.set_title("Improvement vs Spectral Bimodality")
    fig.tight_layout()

    out = Path(save_path)
    fig.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    md_path = str(out) + ".md"
    with open(md_path, "w") as f:
        f.write("# Bimodality vs W1 Improvement Scatter\n\n")
        f.write("Each point is a graph family. x-axis: bimodality index ")
        f.write("(ratio of max to mean band energy, computed from graph topology). ")
        f.write("y-axis: percent W1 improvement from spectral shaping. ")
        f.write("Green = significant (p<0.025). Gray = not significant.\n")
    logger.info(f"Saved {out}.png and {md_path}")
    plt.close(fig)


def print_summary_table(df: pd.DataFrame) -> None:
    """Print markdown summary table to stdout."""
    print("\n## Family Generalization Summary\n")
    print(f"| {'Family':<20} | {'Uniform W1':>12} | {'Spectral W1':>12} | "
          f"{'Improv%':>10} | {'p-value':>10} | {'Significant':>12} |")
    print(f"|{'-'*22}|{'-'*14}|{'-'*14}|{'-'*12}|{'-'*12}|{'-'*14}|")
    for _, row in df.iterrows():
        sig = "YES *" if row["significant"] else "no"
        print(
            f"| {row['family']:<20} "
            f"| {row['uniform_w1_mean']:>8.1f} +/- {row['uniform_w1_std']:<4.1f} "
            f"| {row['spectral_w1_mean']:>8.1f} +/- {row['spectral_w1_std']:<4.1f} "
            f"| {row['improvement_pct']:>9.1f}% "
            f"| {row['p_val']:>10.4f} "
            f"| {sig:>12} |"
        )
    n_sig = df["significant"].sum()
    print(f"\nFamilies significant (p<0.025): {n_sig}/{len(df)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze family generalization results from test_shaping_w1.py"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/diagnostics/family_generalization.json",
        help="Input JSON file from test_shaping_w1.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/diagnostics",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.input)
    print_summary_table(df)
    plot_family_comparison(df, save_path=str(out_dir / "family_comparison"))
    plot_bimodality_correlation(df, save_path=str(out_dir / "family_bimodality_scatter"))

    logger.info("Done.")


if __name__ == "__main__":
    main()
