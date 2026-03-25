"""Analyze scale study results from multiple JSON files.

Loads results/diagnostics/scale_study_n*.json and the baseline n=50 result
(results/diagnostics/shaping_w1_test.json or family_generalization.json),
then produces scaling curve plots and tables.

Usage:
    uv run python scripts/analyze_scale_results.py \
        --result-dir results/diagnostics \
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


def _mean_std_ratio_from_per_seed(per_seed: list[dict]) -> float:
    """Compute mean std_ratio across all seeds and methods."""
    ratios = []
    for seed_data in per_seed:
        for method in ["uniform", "spectral"]:
            sr = seed_data.get(method, {}).get("std_ratio", None)
            if sr is not None:
                ratios.append(sr)
    return float(np.mean(ratios)) if ratios else 0.0


def load_scale_results(result_dir: str) -> pd.DataFrame:
    """Load all scale JSON files plus the n=50 baseline into a DataFrame.

    Globs scale_study_n*.json in result_dir. Also loads shaping_w1_test.json
    or family_generalization.json for the n=50 data point.

    Args:
        result_dir: Directory containing scale_study_n*.json files.

    Returns:
        DataFrame with columns: n_nodes, family, uniform_w1_mean, spectral_w1_mean,
        uniform_w1_std, spectral_w1_std, improvement_pct, p_val, std_ratio_mean.
    """
    result_path = Path(result_dir)
    rows = []

    # Load scale files (n=100, 150, 200, ...)
    scale_files = sorted(result_path.glob("scale_study_n*.json"))
    for fp in scale_files:
        with open(fp) as f:
            data = json.load(f)
        for entry in data:
            rows.append({
                "n_nodes": int(entry["n_nodes"]),
                "family": entry["family"],
                "uniform_w1_mean": entry["uniform_w1_mean"],
                "uniform_w1_std": entry.get("uniform_w1_std", 0.0),
                "spectral_w1_mean": entry["spectral_w1_mean"],
                "spectral_w1_std": entry.get("spectral_w1_std", 0.0),
                "improvement_pct": entry["improvement_pct"],
                "p_val": entry["p_val"],
                "significant": entry.get("significant", False),
                "std_ratio_mean": _mean_std_ratio_from_per_seed(entry.get("per_seed", [])),
                "source_file": fp.name,
            })

    # Load n=50 baseline from shaping_w1_test.json or family_generalization.json
    for baseline_name in ["family_generalization.json", "shaping_w1_test.json"]:
        baseline_path = result_path / baseline_name
        if baseline_path.exists():
            with open(baseline_path) as f:
                data = json.load(f)
            # shaping_w1_test.json has a flat list of per-seed entries
            if isinstance(data, list) and data and "per_seed" not in data[0]:
                # Old flat format: aggregate manually
                import collections
                grouped: dict = collections.defaultdict(lambda: {"uniform": [], "spectral": []})
                for entry in data:
                    grouped[entry["family"]][entry["method"]].append(entry["w1_total"])
                for family, methods in grouped.items():
                    u = np.array(methods["uniform"])
                    s = np.array(methods["spectral"])
                    if len(u) and len(s):
                        rows.append({
                            "n_nodes": 50,
                            "family": family,
                            "uniform_w1_mean": float(u.mean()),
                            "uniform_w1_std": float(u.std()),
                            "spectral_w1_mean": float(s.mean()),
                            "spectral_w1_std": float(s.std()),
                            "improvement_pct": float((u.mean() - s.mean()) / u.mean() * 100)
                                               if u.mean() > 0 else 0.0,
                            "p_val": 1.0,
                            "significant": False,
                            "std_ratio_mean": 0.0,
                            "source_file": baseline_name,
                        })
            else:
                # New format from test_shaping_w1.py with run_family()
                for entry in data:
                    if entry.get("n_nodes", 50) == 50:
                        rows.append({
                            "n_nodes": 50,
                            "family": entry["family"],
                            "uniform_w1_mean": entry["uniform_w1_mean"],
                            "uniform_w1_std": entry.get("uniform_w1_std", 0.0),
                            "spectral_w1_mean": entry["spectral_w1_mean"],
                            "spectral_w1_std": entry.get("spectral_w1_std", 0.0),
                            "improvement_pct": entry["improvement_pct"],
                            "p_val": entry["p_val"],
                            "significant": entry.get("significant", False),
                            "std_ratio_mean": _mean_std_ratio_from_per_seed(
                                entry.get("per_seed", [])
                            ),
                            "source_file": baseline_name,
                        })
            break  # use first match

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No scale results found in %s", result_dir)
        return df

    df = df.sort_values(["family", "n_nodes"]).reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows from {len(scale_files)} scale files + baseline")
    return df


def plot_scaling_curve(df: pd.DataFrame, save_path: str) -> None:
    """Line plot of improvement_pct vs n_nodes per family.

    Includes shaded 1-std band. Marks capacity limit (first n_nodes where
    std_ratio_mean > 3) with a vertical dashed line.

    Args:
        df: DataFrame from load_scale_results().
        save_path: Path (without extension) to save the PNG.
    """
    families = df["family"].unique()
    colors = sns.color_palette("husl", len(families))
    fig, ax = plt.subplots(figsize=(10, 6))

    capacity_limits = []
    for family, color in zip(families, colors):
        fam = df[df["family"] == family].sort_values("n_nodes")
        if fam.empty:
            continue
        x = fam["n_nodes"].values
        y = fam["improvement_pct"].values

        # Use per-seed std of improvement_pct if available; else use propagated std
        u_std = fam["uniform_w1_std"].values
        s_std = fam["spectral_w1_std"].values
        u_mean = fam["uniform_w1_mean"].values
        # Approximate std of improvement_pct via error propagation
        improv_std = np.sqrt(u_std**2 + s_std**2) / np.maximum(u_mean, 1e-8) * 100

        ax.plot(x, y, marker="o", color=color, label=family)
        ax.fill_between(x, y - improv_std, y + improv_std, alpha=0.2, color=color)

        # Find capacity limit
        limit_mask = fam["std_ratio_mean"].values > 3
        if limit_mask.any():
            limit_n = fam["n_nodes"].values[limit_mask][0]
            capacity_limits.append(limit_n)

    if capacity_limits:
        limit_n = min(capacity_limits)
        ax.axvline(x=limit_n, color="red", linestyle="--", alpha=0.7,
                   label=f"Capacity limit (n={limit_n}, std_ratio>3)")

    ax.axhline(y=0, color="black", lw=0.8, linestyle=":")
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("W1 Improvement (%)")
    ax.set_title("Scaling Curve: W1 Improvement vs Graph Size")
    ax.legend()
    fig.tight_layout()

    out = Path(save_path)
    fig.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    md_path = str(out) + ".md"
    with open(md_path, "w") as f:
        f.write("# Scaling Curve: W1 Improvement vs Graph Size\n\n")
        f.write("Line plot of W1 improvement percentage vs number of nodes per family. ")
        f.write("Shaded band = propagated 1-std. Dashed red line = capacity limit "
                "(first n where mean std_ratio > 3).\n")
    logger.info(f"Saved {out}.png and {md_path}")
    plt.close(fig)


def plot_w1_vs_scale(df: pd.DataFrame, save_path: str) -> None:
    """Raw W1 (both methods) vs n_nodes per family.

    Args:
        df: DataFrame from load_scale_results().
        save_path: Path (without extension) to save the PNG.
    """
    families = df["family"].unique()
    fig, axes = plt.subplots(1, len(families), figsize=(7 * len(families), 5), squeeze=False)

    for ax, family in zip(axes[0], families):
        fam = df[df["family"] == family].sort_values("n_nodes")
        if fam.empty:
            continue
        x = fam["n_nodes"].values
        ax.errorbar(x, fam["uniform_w1_mean"], yerr=fam["uniform_w1_std"],
                    marker="o", label="Uniform", color="steelblue", capsize=4)
        ax.errorbar(x, fam["spectral_w1_mean"], yerr=fam["spectral_w1_std"],
                    marker="s", label="Spectral", color="coral", capsize=4)
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel("Total W1 Distance")
        ax.set_title(family)
        ax.legend()

    fig.suptitle("Raw W1 Distance vs Scale", y=1.02)
    fig.tight_layout()

    out = Path(save_path)
    fig.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    md_path = str(out) + ".md"
    with open(md_path, "w") as f:
        f.write("# Raw W1 Distance vs Scale\n\n")
        f.write("Per-family W1 distance for uniform and spectral noise across graph sizes.\n")
    logger.info(f"Saved {out}.png and {md_path}")
    plt.close(fig)


def print_scale_table(df: pd.DataFrame, output_dir: str) -> None:
    """Print and save a markdown table of scale results."""
    lines = ["# Scale Study Results\n",
             f"| {'Family':<20} | {'n_nodes':>7} | {'Uniform W1':>12} | {'Spectral W1':>12} | "
             f"{'Improv%':>10} | {'p-val':>8} | {'std_ratio':>10} |",
             f"|{'-'*22}|{'-'*9}|{'-'*14}|{'-'*14}|{'-'*12}|{'-'*10}|{'-'*12}|"]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['family']:<20} "
            f"| {int(row['n_nodes']):>7} "
            f"| {row['uniform_w1_mean']:>8.1f}+/-{row['uniform_w1_std']:<4.1f} "
            f"| {row['spectral_w1_mean']:>8.1f}+/-{row['spectral_w1_std']:<4.1f} "
            f"| {row['improvement_pct']:>9.1f}% "
            f"| {row['p_val']:>8.4f} "
            f"| {row['std_ratio_mean']:>10.2f} |"
        )
    table_text = "\n".join(lines)
    print(table_text)

    out_path = Path(output_dir) / "scale_table.md"
    with open(out_path, "w") as f:
        f.write(table_text + "\n")
    logger.info(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze scale study results"
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="results/diagnostics",
        help="Directory containing scale_study_n*.json files",
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

    df = load_scale_results(args.result_dir)
    if df.empty:
        logger.error("No data loaded. Run run_scale_study.sh first.")
        return

    print_scale_table(df, args.output_dir)
    plot_scaling_curve(df, save_path=str(out_dir / "scaling_curve"))
    plot_w1_vs_scale(df, save_path=str(out_dir / "w1_vs_scale"))

    logger.info("Done.")


if __name__ == "__main__":
    main()
