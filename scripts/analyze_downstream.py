"""Analyze downstream node classification results from run_downstream.py.

Loads results/diagnostics/downstream_results.json and produces:
  - Summary markdown table
  - Bar chart comparing uniform vs spectral classification accuracy per family

Usage:
    uv run python scripts/analyze_downstream.py \
        --input results/diagnostics/downstream_results.json \
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def load_results(path: str) -> pd.DataFrame:
    """Load downstream JSON into a summary DataFrame."""
    with open(path) as f:
        data = json.load(f)

    rows = []
    for entry in data:
        rows.append({
            "family": entry["family"],
            "n_nodes": entry.get("n_nodes", 50),
            "uniform_acc_mean": entry["uniform_acc_mean"],
            "uniform_acc_std": entry.get("uniform_acc_std", 0.0),
            "spectral_acc_mean": entry["spectral_acc_mean"],
            "spectral_acc_std": entry.get("spectral_acc_std", 0.0),
            "improvement_pct": entry["improvement_pct"],
            "p_val": entry["p_val"],
            "significant": entry.get("significant", False),
            "n_communities": entry.get("n_communities", 4),
        })
    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} family downstream results from {path}")
    return df


def plot_accuracy_comparison(df: pd.DataFrame, save_path: str) -> None:
    """Grouped bar chart: uniform vs spectral accuracy per family.

    Args:
        df: DataFrame from load_results().
        save_path: Path (without extension) to save the PNG.
    """
    families = df["family"].tolist()
    x = np.arange(len(families))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width / 2, df["uniform_acc_mean"], width,
           yerr=df["uniform_acc_std"], label="Uniform",
           color="steelblue", alpha=0.85, capsize=4)
    ax.bar(x + width / 2, df["spectral_acc_mean"], width,
           yerr=df["spectral_acc_std"], label="Graph-FANS (spectral)",
           color="coral", alpha=0.85, capsize=4)

    # Add random-chance baseline
    for i, (_, row) in enumerate(df.iterrows()):
        random_acc = 1.0 / max(row["n_communities"], 1)
        ax.axhline(y=random_acc, color="gray", linestyle=":", alpha=0.5, lw=1)

    # Significance stars
    for i, (_, row) in enumerate(df.iterrows()):
        if row["significant"]:
            y_top = row["spectral_acc_mean"] + row["spectral_acc_std"] + 0.02
            ax.text(i + width / 2, y_top, "*", ha="center", va="bottom",
                    fontsize=14, color="green", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=15, ha="right")
    ax.set_ylabel("Node Classification Accuracy")
    ax.set_ylim(0, min(1.05, df[["uniform_acc_mean", "spectral_acc_mean"]].max().max() + 0.15))
    ax.set_title("Downstream: Node Classification Accuracy")
    ax.legend()
    fig.tight_layout()

    out = Path(save_path)
    fig.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    md_path = str(out) + ".md"
    with open(md_path, "w") as f:
        f.write("# Downstream: Node Classification Accuracy\n\n")
        f.write("Classifier trained on generated features, tested on reference features. ")
        f.write("Dotted line = random-chance baseline (1/n_communities). ")
        f.write("* = significant improvement (p<0.05).\n")
    logger.info(f"Saved {out}.png and {md_path}")
    plt.close(fig)


def print_summary_table(df: pd.DataFrame, output_dir: str | None = None) -> None:
    """Print markdown summary table."""
    lines = ["## Downstream Evaluation Summary\n",
             f"| {'Family':<20} | {'Uniform Acc':>14} | {'Spectral Acc':>14} | "
             f"{'Improv%':>10} | {'p-value':>10} |",
             f"|{'-'*22}|{'-'*16}|{'-'*16}|{'-'*12}|{'-'*12}|"]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['family']:<20} "
            f"| {row['uniform_acc_mean']:>8.3f}+/-{row['uniform_acc_std']:<5.3f} "
            f"| {row['spectral_acc_mean']:>8.3f}+/-{row['spectral_acc_std']:<5.3f} "
            f"| {row['improvement_pct']:>9.1f}% "
            f"| {row['p_val']:>10.4f} |"
        )
    text = "\n".join(lines)
    print(text)

    if output_dir:
        out_path = Path(output_dir) / "downstream_table.md"
        with open(out_path, "w") as f:
            f.write(text + "\n")
        logger.info(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze downstream evaluation results"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/diagnostics/downstream_results.json",
        help="Input JSON from run_downstream.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/diagnostics",
        help="Output directory for plots and table",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.input)
    print_summary_table(df, output_dir=args.output_dir)
    plot_accuracy_comparison(df, save_path=str(out_dir / "downstream_accuracy"))

    logger.info("Done.")


if __name__ == "__main__":
    main()
