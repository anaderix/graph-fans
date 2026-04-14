"""InfoGrid: non-uniform DDIM timestep spacing based on entropy rate.

Concentrates DDIM denoising steps in the "informative window" — the region
of sigma-space (equivalently t-space) where the entropy rate is highest.
This should improve generation quality and/or efficiency vs uniform spacing.

Key design choices:
- Builds a cumulative information coordinate u(sigma) via trapezoidal integration.
- Spaces n_steps points uniformly in u-space, then maps back to t-space.
- Returns timesteps from T to 0 (decreasing), matching trainer.generate() convention.
"""

from __future__ import annotations

import logging

import numpy as np

from .info_noise import sigma_to_t

logger = logging.getLogger(__name__)


def build_info_grid(
    entropy_rate_profile: dict,
    sde,
    n_steps: int = 200,
) -> np.ndarray:
    """Build non-uniform DDIM timestep grid concentrated in high-entropy regions.

    1. Read r_hat(sigma) from the entropy rate profile.
    2. Build cumulative information coordinate u(sigma) via trapezoidal integration.
    3. Space n_steps+1 points uniformly in u-space.
    4. Map back to sigma-space via linear interpolation of the inverse CDF.
    5. Convert to t-space.

    Args:
        entropy_rate_profile: Dict from get_entropy_rate_profile() with keys
            'sigma_centers' and 'entropy_rates'.
        sde: SDE instance with marginal_params() and T attribute.
        n_steps: Number of DDIM steps (returns n_steps+1 timesteps).

    Returns:
        Array of shape [n_steps + 1] with timesteps from T to 0 (decreasing).
    """
    sigmas = np.array(entropy_rate_profile["sigma_centers"])
    rates = np.array(entropy_rate_profile["entropy_rates"])

    # Floor: ensure no zero rates (prevents collapsed grid regions)
    floor = max(rates.max() * 0.01, 1e-10)
    rates_floored = np.maximum(rates, floor)

    # Sort by sigma (should already be sorted, but ensure)
    order = np.argsort(sigmas)
    sigmas = sigmas[order]
    rates_floored = rates_floored[order]

    # Trapezoidal cumulative integration: u(sigma)
    # u[i] = integral from sigma[0] to sigma[i] of r(sigma') dsigma'
    u = np.zeros(len(sigmas))
    for i in range(1, len(sigmas)):
        ds = sigmas[i] - sigmas[i - 1]
        u[i] = u[i - 1] + 0.5 * (rates_floored[i - 1] + rates_floored[i]) * ds

    # Normalize to [0, 1]
    u_total = u[-1]
    if u_total < 1e-15:
        logger.warning("InfoGrid: zero total information, falling back to uniform grid")
        return build_uniform_grid(sde, n_steps)

    u_norm = u / u_total

    # Space n_steps+1 points uniformly in u-space (including endpoints)
    u_targets = np.linspace(0.0, 1.0, n_steps + 1)

    # Map back to sigma-space via linear interpolation of the inverse CDF
    sigma_grid = np.interp(u_targets, u_norm, sigmas)

    # Convert sigma to t
    t_grid = np.array([sigma_to_t(sde, float(s)) for s in sigma_grid])

    # Ensure grid goes from T to 0 (decreasing)
    # sigma increases with t, so t_grid is increasing — reverse it
    t_grid = np.sort(t_grid)[::-1]

    # Clamp endpoints: start at T, end at 0
    t_grid[0] = sde.T
    t_grid[-1] = 0.0

    logger.debug(
        f"InfoGrid: {n_steps} steps, t=[{t_grid[0]:.4f}, ..., {t_grid[-1]:.4f}], "
        f"median_dt={np.median(np.abs(np.diff(t_grid))):.4f}"
    )

    return t_grid


def build_uniform_grid(sde, n_steps: int = 200) -> np.ndarray:
    """Build a uniform DDIM timestep grid from T to 0.

    This matches the default behavior in trainer.generate().

    Args:
        sde: SDE instance with T attribute.
        n_steps: Number of DDIM steps.

    Returns:
        Array of shape [n_steps + 1] with timesteps from T to 0 (decreasing).
    """
    return np.linspace(sde.T, 0, n_steps + 1)


def visualize_grids(
    uniform_grid: np.ndarray,
    info_grid: np.ndarray,
    entropy_rate_profile: dict,
    save_path: str | None = None,
) -> None:
    """Plot uniform vs InfoGrid step placement overlaid on entropy rate.

    Args:
        uniform_grid: Uniform timestep grid [n_steps+1].
        info_grid: InfoGrid timestep grid [n_steps+1].
        entropy_rate_profile: Dict from get_entropy_rate_profile().
        save_path: If provided, save figure to this path.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping grid visualization")
        return

    sigmas = np.array(entropy_rate_profile["sigma_centers"])
    rates = np.array(entropy_rate_profile["entropy_rates"])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    # Top: entropy rate profile
    ax = axes[0]
    ax.plot(sigmas, rates, "b-", linewidth=2, label="Entropy rate")
    ax.set_xlabel("sigma")
    ax.set_ylabel("r_hat(sigma)")
    ax.set_title("Entropy Rate Profile")
    ax.legend()
    ax.set_xscale("log")

    # Bottom: step density comparison
    ax = axes[1]
    # Convert grids to sigma for comparison
    # Use t values directly
    ax.hist(
        uniform_grid[:-1], bins=30, alpha=0.5, density=True,
        label="Uniform grid", color="gray",
    )
    ax.hist(
        info_grid[:-1], bins=30, alpha=0.5, density=True,
        label="InfoGrid", color="blue",
    )
    ax.set_xlabel("t")
    ax.set_ylabel("Step density")
    ax.set_title("DDIM Step Placement")
    ax.legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Grid visualization saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
