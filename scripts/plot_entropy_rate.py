"""Plot entropy rate profiles from Phase 4a.

Generates diagnostic visualizations:
1. Entropy rate profile r_hat(sigma) with informative window annotation
2. Sampling density comparison (uniform vs log-SNR vs InfoNoise)
3. Per-bin loss comparison

Usage:
    uv run python scripts/plot_entropy_rate.py \\
        --profile results/phase4a/entropy_rate_profiles/SBM_q0.05_seed0_info_noise.json \\
        --output results/phase4a/entropy_rate_plot.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def plot_entropy_rate(
    profile: dict,
    title: str = "Entropy Rate Profile",
    save_path: str | None = None,
) -> None:
    """Plot entropy rate profile with informative window annotation."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib not available. Install with: uv add matplotlib")
        return

    sigmas = np.array(profile["sigma_centers"])
    rates = np.array(profile["entropy_rates"])
    ema_losses = np.array(profile["ema_losses"])
    counts = np.array(profile["counts"])

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # 1. Entropy rate profile
    ax = axes[0]
    ax.plot(sigmas, rates, "b-", linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("sigma (noise std)")
    ax.set_ylabel("r_hat(sigma)")
    ax.set_title(f"{title}: Entropy Rate")
    ax.set_xscale("log")

    # Annotate informative window (top 50% of entropy rate)
    if rates.max() > 0:
        threshold = rates.max() * 0.5
        in_window = rates >= threshold
        if in_window.any():
            window_sigmas = sigmas[in_window]
            ax.axvspan(
                window_sigmas.min(), window_sigmas.max(),
                alpha=0.2, color="green", label="Informative window (>50% peak)",
            )
            ax.legend()

    # 2. Per-bin EMA loss
    ax = axes[1]
    ax.bar(range(len(sigmas)), ema_losses, color="orange", alpha=0.7)
    ax.set_xlabel("Bin index")
    ax.set_ylabel("EMA loss")
    ax.set_title("Per-Bin EMA Loss")

    # Add sigma labels on top
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_positions = list(range(0, len(sigmas), max(1, len(sigmas) // 5)))
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([f"{sigmas[i]:.3f}" for i in tick_positions], fontsize=8)
    ax2.set_xlabel("sigma")

    # 3. Observation counts per bin
    ax = axes[2]
    ax.bar(range(len(sigmas)), counts, color="gray", alpha=0.7)
    ax.set_xlabel("Bin index")
    ax.set_ylabel("Observation count")
    ax.set_title("Per-Bin Observation Count")

    # Annotate total steps
    ax.text(
        0.02, 0.95, f"Total steps: {profile['step_count']}",
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_sampling_density(
    profile: dict,
    sde=None,
    title: str = "Sampling Density Comparison",
    save_path: str | None = None,
) -> None:
    """Plot sampling density comparison: uniform vs InfoNoise."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib not available")
        return

    from graph_fans.phase2.info_noise import (
        InfoNoiseState,
        build_sampler_cdf,
        create_info_noise_state,
        record_observation,
    )
    from graph_fans.phase2.sde import CosineScheduleSDE

    if sde is None:
        sde = CosineScheduleSDE()

    sigmas = np.array(profile["sigma_centers"])
    rates = np.array(profile["entropy_rates"])

    # Reconstruct CDF from rates for visualization
    floor = max(rates.max() * 0.01, 1e-10)
    rates_floored = np.maximum(rates, floor)
    pdf = rates_floored / rates_floored.sum()

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Uniform density
    uniform_pdf = np.ones(len(sigmas)) / len(sigmas)
    ax.plot(sigmas, uniform_pdf, "r--", linewidth=2, label="Uniform (in sigma)")
    ax.plot(sigmas, pdf, "b-", linewidth=2, label="InfoNoise adaptive")
    ax.fill_between(sigmas, pdf, alpha=0.2, color="blue")

    ax.set_xlabel("sigma (noise std)")
    ax.set_ylabel("Sampling density (normalized)")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Sampling density plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot entropy rate profiles from Phase 4a"
    )
    parser.add_argument(
        "--profile", type=str, required=True,
        help="Path to entropy rate profile JSON",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output PNG path (if not set, shows interactively)",
    )
    parser.add_argument(
        "--density-output", type=str, default=None,
        help="Output PNG path for sampling density plot",
    )
    parser.add_argument(
        "--title", type=str, default="Entropy Rate Profile",
        help="Plot title prefix",
    )
    args = parser.parse_args()

    profile_path = Path(args.profile)
    if not profile_path.exists():
        logger.error(f"Profile not found: {profile_path}")
        sys.exit(1)

    with open(profile_path) as f:
        profile = json.load(f)

    logger.info(f"Loaded profile: {len(profile['sigma_centers'])} bins, "
                f"{profile['step_count']} steps")

    plot_entropy_rate(profile, title=args.title, save_path=args.output)

    if args.density_output is not None:
        plot_sampling_density(
            profile, title=f"{args.title}: Sampling Density",
            save_path=args.density_output,
        )


if __name__ == "__main__":
    main()
