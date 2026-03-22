"""Visualization functions for Phase 0 spectral profiling."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .spectral_profiler import SpectralProfile
from graph_fans.utils.save import save_figure

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def plot_spectral_profile(
    profile: SpectralProfile, ax: plt.Axes | None = None
) -> plt.Axes:
    """Bar chart of per-band energy with band boundaries labeled."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    B = len(profile.band_energies)
    x = np.arange(B)
    colors = sns.color_palette("viridis", B)

    ax.bar(x, profile.normalized_band_energies, color=colors, edgecolor="white", lw=0.5)

    labels = [f"[{lo:.2f},{hi:.2f})" for lo, hi in profile.band_boundaries]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Normalized Energy")
    ax.set_title(f"{profile.name}\n(ratio={profile.energy_ratio:.1f}×, {profile.decay_model})")
    ax.set_xlabel("Eigenvalue Band")

    return ax


def plot_all_profiles(
    profiles: dict[str, SpectralProfile], save_path: str | Path | None = None
) -> plt.Figure:
    """Grid of spectral profiles for all graph families."""
    n = len(profiles)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, (name, profile) in zip(axes_flat, profiles.items()):
        plot_spectral_profile(profile, ax=ax)

    # Hide unused axes
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Spectral Energy Profiles", fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path, "Spectral Energy Profiles")

    return fig


def plot_energy_decay(
    profiles: dict[str, SpectralProfile], save_path: str | Path | None = None
) -> plt.Figure:
    """Overlay plot of normalized band energy curves with fitted decay models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("husl", len(profiles))

    for (name, profile), color in zip(profiles.items(), colors):
        B = len(profile.normalized_band_energies)
        x = np.arange(B)
        ax.plot(
            x,
            profile.normalized_band_energies,
            "o-",
            color=color,
            label=f"{name} (ratio={profile.energy_ratio:.1f}×)",
            markersize=5,
        )

    ax.set_xlabel("Band Index (low → high eigenvalue)")
    ax.set_ylabel("Normalized Energy")
    ax.set_title("Spectral Energy Decay Across Graph Families")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path, "Spectral Energy Decay Across Graph Families")

    return fig


def plot_g0_summary(
    profiles: dict[str, SpectralProfile], save_path: str | Path | None = None
) -> plt.Figure:
    """Summary figure with energy ratios, decay parameters, and Go/No-Go assessment."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    names = list(profiles.keys())
    ratios = [profiles[n].energy_ratio for n in names]
    colors = ["green" if r >= 2.0 else "red" for r in ratios]

    # Energy ratio bar chart
    ax1.barh(names, ratios, color=colors, edgecolor="white")
    ax1.axvline(x=2.0, color="black", linestyle="--", lw=1.5, label="G0 threshold (2×)")
    ax1.set_xlabel("Energy Ratio (max/min band)")
    ax1.set_title("G0 Gate: Non-Uniform Spectral Energy")
    ax1.legend()

    # Decay model summary
    decay_data = []
    for name in names:
        p = profiles[name]
        decay_data.append(
            f"{p.decay_model}\nR²={p.decay_r_squared:.2f}"
        )
    ax2.barh(names, [profiles[n].decay_r_squared for n in names], color="steelblue")
    ax2.set_xlabel("Decay Fit R²")
    ax2.set_title("Spectral Energy Decay Quality")

    # Annotate with model type
    for i, txt in enumerate(decay_data):
        ax2.text(0.02, i, txt, va="center", fontsize=8, color="white", fontweight="bold")

    # G0 decision
    passing = sum(1 for r in ratios if r >= 2.0)
    decision = "GO" if passing >= 2 else "NO-GO"
    decision_color = "green" if decision == "GO" else "red"
    fig.text(
        0.5, -0.05,
        f"G0 Decision: {decision} ({passing}/{len(ratios)} families with ratio ≥ 2×)",
        ha="center", fontsize=14, fontweight="bold", color=decision_color,
    )

    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path, "G0 Gate Summary")

    return fig
