"""Visualization functions for Phase 1a: Spectral Metrics Validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

from .spectral_metrics import heat_kernel_trace
from graph_fans.utils.save import save_figure

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def plot_metric_comparison(
    results_df: pd.DataFrame, save_path: str | Path | None = None
) -> plt.Figure:
    """Grouped bar chart comparing standard vs spectral metrics."""
    fig, ax = plt.subplots(figsize=(12, 6))

    spectral_cols = ["Spectral JSD", "QBE Distance", "HKS Distance"]
    standard_cols = ["Degree MMD", "Clustering MMD", "Triangle MMD"]

    labels = [
        f"{row['Reference']}\nvs {row['Generated']}"
        for _, row in results_df.iterrows()
    ]
    x = np.arange(len(labels))
    width = 0.12

    for i, col in enumerate(standard_cols):
        ax.bar(x - 0.21 + i * width, results_df[col], width, label=col, alpha=0.8)
    for i, col in enumerate(spectral_cols):
        ax.bar(
            x + 0.21 + i * width,
            results_df[col],
            width,
            label=col,
            alpha=0.8,
            hatch="//",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Metric Value")
    ax.set_title("Standard vs Spectral Metrics Comparison")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path, "Standard vs Spectral Metrics Comparison")

    return fig


def plot_spectral_density_comparison(
    eigenvalues_ref: np.ndarray,
    eigenvalues_gen: np.ndarray,
    title: str = "",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Overlay KDE plots of eigenvalue densities."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ev_ref = eigenvalues_ref / (eigenvalues_ref.max() + 1e-10)
    ev_gen = eigenvalues_gen / (eigenvalues_gen.max() + 1e-10)

    ev_ref = ev_ref[ev_ref > 1e-8]
    ev_gen = ev_gen[ev_gen > 1e-8]

    grid = np.linspace(0, 1, 500)

    if len(ev_ref) >= 2:
        kde_ref = gaussian_kde(ev_ref, bw_method="silverman")
        ax.plot(grid, kde_ref(grid), label="Reference", lw=2)
        ax.fill_between(grid, kde_ref(grid), alpha=0.2)

    if len(ev_gen) >= 2:
        kde_gen = gaussian_kde(ev_gen, bw_method="silverman")
        ax.plot(grid, kde_gen(grid), label="Generated", lw=2, linestyle="--")
        ax.fill_between(grid, kde_gen(grid), alpha=0.2)

    ax.set_xlabel("Normalized Eigenvalue (λ/λ_max)")
    ax.set_ylabel("Density")
    ax.set_title(f"Spectral Density Comparison{': ' + title if title else ''}")
    ax.legend()
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path, f"Spectral Density: {title}" if title else "Spectral Density Comparison")

    return fig


def plot_hks_comparison(
    eigenvalues_ref: np.ndarray,
    eigenvalues_gen: np.ndarray,
    t_values: np.ndarray | None = None,
    title: str = "",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """HKS trace curves comparison."""
    if t_values is None:
        t_values = np.logspace(-2, 2, 50)

    fig, ax = plt.subplots(figsize=(8, 5))

    hks_ref = heat_kernel_trace(eigenvalues_ref, t_values) / len(eigenvalues_ref)
    hks_gen = heat_kernel_trace(eigenvalues_gen, t_values) / len(eigenvalues_gen)

    ax.plot(t_values, hks_ref, "o-", label="Reference", markersize=3, lw=1.5)
    ax.plot(t_values, hks_gen, "s--", label="Generated", markersize=3, lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Diffusion Time t")
    ax.set_ylabel("Normalized Tr(exp(-tL)) / n")
    ax.set_title(f"Heat Kernel Signature{': ' + title if title else ''}")
    ax.legend()
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path, f"HKS: {title}" if title else "Heat Kernel Signature")

    return fig


def plot_information_gap(
    results_df: pd.DataFrame, save_path: str | Path | None = None
) -> plt.Figure:
    """Scatter plot of spectral aggregate vs standard aggregate showing the gap."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        results_df["Standard Aggregate"],
        results_df["Spectral Aggregate"],
        s=100, c="steelblue", edgecolor="white", zorder=3,
    )

    for _, row in results_df.iterrows():
        ax.annotate(
            f"{row['Reference']}\nvs {row['Generated']}",
            (row["Standard Aggregate"], row["Spectral Aggregate"]),
            fontsize=7, ha="left", va="bottom",
            xytext=(5, 5), textcoords="offset points",
        )

    # Diagonal line (perfect agreement)
    lims = [0, max(results_df["Standard Aggregate"].max(), results_df["Spectral Aggregate"].max()) * 1.1]
    ax.plot(lims, lims, "k--", alpha=0.5, label="Perfect agreement")

    ax.set_xlabel("Standard Aggregate (MMD)")
    ax.set_ylabel("Spectral Aggregate")
    ax.set_title("Information Gap: Spectral vs Standard Metrics")
    ax.legend()
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path, "Information Gap: Spectral vs Standard Metrics")

    return fig


def plot_g1_summary(
    results_df: pd.DataFrame,
    g1_result: dict | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Summary figure for G1 decision."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: metric comparison heatmap
    metric_cols = [
        "Spectral JSD", "QBE Distance", "HKS Distance",
        "Degree MMD", "Clustering MMD", "Triangle MMD",
    ]
    available = [c for c in metric_cols if c in results_df.columns]
    labels = [
        f"{row['Reference']} vs {row['Generated']}"
        for _, row in results_df.iterrows()
    ]

    heatmap_data = results_df[available].copy()
    heatmap_data.index = labels
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax1)
    ax1.set_title("All Metrics Heatmap")

    # Right: gap visualization
    if "Spectral Aggregate" in results_df.columns and "Standard Aggregate" in results_df.columns:
        gaps = abs(results_df["Spectral Aggregate"] - results_df["Standard Aggregate"])
        rel_gaps = gaps / (results_df["Standard Aggregate"] + 1e-10)
        colors = ["green" if g >= 0.10 else "red" for g in rel_gaps]
        ax2.barh(labels, rel_gaps, color=colors)
        ax2.axvline(x=0.10, color="black", linestyle="--", lw=1.5, label="10% threshold")
        ax2.set_xlabel("Relative Gap")
        ax2.set_title("G1 Gate: Spectral Information Gap")
        ax2.legend()

    if g1_result:
        decision = g1_result.get("decision", "UNKNOWN")
        color = "green" if decision == "GO" else "red"
        fig.text(
            0.5, -0.05,
            f"G1 Decision: {decision}",
            ha="center", fontsize=14, fontweight="bold", color=color,
        )

    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path, "G1 Gate Summary")

    return fig
