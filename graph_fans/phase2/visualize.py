"""Visualization functions for Phase 2."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from graph_fans.utils.save import save_figure

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def plot_per_band_comparison(
    df: pd.DataFrame, B: int = 8, save_path: str | Path | None = None
) -> plt.Figure:
    """Per-band QBE comparison: uniform vs spectral, grouped by family."""
    families = df["family"].unique()
    fig, axes = plt.subplots(1, len(families), figsize=(6 * len(families), 5), squeeze=False)

    band_cols = [f"qbe_band_{b}" for b in range(B)]

    for ax, family in zip(axes[0], families):
        for method, color, label in [
            ("uniform", "steelblue", "Uniform"),
            ("spectral", "coral", "Graph-FANS"),
        ]:
            subset = df[(df["family"] == family) & (df["method"] == method)]
            if subset.empty:
                continue
            means = subset[band_cols].mean().values
            stds = subset[band_cols].std().values
            x = np.arange(B)
            width = 0.35
            offset = -width / 2 if method == "uniform" else width / 2
            ax.bar(x + offset, means, width, yerr=stds, label=label,
                   color=color, alpha=0.8, capsize=2)

        ax.set_xlabel("Band (low → high eigenvalue)")
        ax.set_ylabel("QBE Difference")
        ax.set_title(family)
        ax.legend()
        ax.set_xticks(range(B))

    fig.suptitle("Per-Band Spectral Fidelity: Uniform vs Graph-FANS", y=1.02)
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path, "Per-Band Spectral Fidelity Comparison")
    return fig


def plot_h1a_summary(
    df: pd.DataFrame, save_path: str | Path | None = None
) -> plt.Figure:
    """Summary table + chart for H1-A results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    families = df["family"].unique()

    # Left: QBE high-band comparison
    data = []
    for family in families:
        for method in ["uniform", "spectral"]:
            subset = df[(df["family"] == family) & (df["method"] == method)]
            if not subset.empty:
                data.append({
                    "Family": family,
                    "Method": method.title(),
                    "QBE (high bands)": subset["qbe_high_bands"].mean(),
                    "std": subset["qbe_high_bands"].std(),
                })

    plot_df = pd.DataFrame(data)
    if not plot_df.empty:
        x = np.arange(len(families))
        width = 0.35
        uniform_vals = plot_df[plot_df["Method"] == "Uniform"]["QBE (high bands)"].values
        spectral_vals = plot_df[plot_df["Method"] == "Spectral"]["QBE (high bands)"].values
        uniform_err = plot_df[plot_df["Method"] == "Uniform"]["std"].values
        spectral_err = plot_df[plot_df["Method"] == "Spectral"]["std"].values

        ax1.bar(x - width / 2, uniform_vals, width, yerr=uniform_err,
                label="Uniform", color="steelblue", capsize=3)
        ax1.bar(x + width / 2, spectral_vals, width, yerr=spectral_err,
                label="Graph-FANS", color="coral", capsize=3)
        ax1.set_xticks(x)
        ax1.set_xticklabels(families)
        ax1.set_ylabel("QBE Distance (high bands)")
        ax1.set_title("H1-A: High-Band Fidelity")
        ax1.legend()

    # Right: improvement percentage
    improvements = []
    for family in families:
        u = df[(df["family"] == family) & (df["method"] == "uniform")]["qbe_high_bands"].mean()
        s = df[(df["family"] == family) & (df["method"] == "spectral")]["qbe_high_bands"].mean()
        if u > 0:
            improvements.append({"Family": family, "Improvement (%)": (u - s) / u * 100})
    if improvements:
        imp_df = pd.DataFrame(improvements)
        colors = ["green" if v > 0 else "red" for v in imp_df["Improvement (%)"]]
        ax2.barh(imp_df["Family"], imp_df["Improvement (%)"], color=colors)
        ax2.axvline(x=0, color="black", lw=0.5)
        ax2.set_xlabel("Improvement (%)")
        ax2.set_title("Graph-FANS vs Uniform (high bands)")

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, "H1-A Summary")
    return fig


def plot_h1a_decision_summary(
    g2_decision: dict, save_path: str | Path | None = None
) -> plt.Figure:
    """H1-A decision summary: per-family improvement bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    h1a = g2_decision["h1a"]
    families = list(h1a["details"].keys())
    improvements = [h1a["details"][f]["improvement"] for f in families]
    colors = ["green" if v > 0 else "red" for v in improvements]
    ax.barh(families, improvements, color=colors)
    ax.axvline(x=0, color="black", lw=1)
    ax.set_xlabel("QBE Improvement (uniform − spectral)")
    ax.set_title(
        f"H1-A: {'PASS' if h1a['pass'] else 'FAIL'} "
        f"({h1a['families_with_improvement']}/{h1a['total_families']} families improved)"
    )

    decision = g2_decision["decision"]
    color = "green" if decision == "GO" else "red"
    fig.text(
        0.5, -0.05, f"Decision: {decision}",
        ha="center", fontsize=14, fontweight="bold", color=color,
    )

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, "H1-A Decision Summary")
    return fig
