"""InfoNoise: online entropy-rate estimator and adaptive timestep sampler.

Implements the InfoNoise approach for data-adaptive timestep sampling in
diffusion training. The entropy rate r_hat(sigma) = mmse_hat(sigma) / sigma^3
identifies the "informative window" where the model benefits most from training.

Key design choices:
- Uses sigma space (not t space) because entropy rate is naturally defined there.
- Bins are uniform in log-sigma space for even coverage across noise levels.
- FIFO buffer + EMA smoothing per bin for online estimation.
- Gated regularization near sigma_min prevents boundary artifacts.
- Warm-up period samples uniformly in t before switching to adaptive sampling.

Follows the dataclass + pure functions pattern from noise_shaper.py.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EntropyRateBin:
    """Single bin in the entropy rate histogram."""

    sigma_center: float
    losses: deque  # FIFO buffer of MSE losses (capacity = buffer_capacity)
    ema_loss: float = 0.0
    count: int = 0


@dataclass
class InfoNoiseState:
    """State for the online entropy rate estimator and adaptive sampler."""

    bins: list[EntropyRateBin]
    n_bins: int
    buffer_capacity: int
    ema_alpha: float
    sigma_min: float
    sigma_max: float
    warm_up_steps: int
    refresh_interval: int
    sigma_min_gate_width: float
    _step_count: int = 0
    _cdf: np.ndarray | None = None
    # Log-sigma bin edges for routing observations
    _log_sigma_edges: np.ndarray = field(default_factory=lambda: np.array([]))


def create_info_noise_state(
    sde,
    n_bins: int = 20,
    buffer_capacity: int = 256,
    ema_alpha: float = 0.01,
    warm_up_steps: int = 2000,
    refresh_interval: int = 500,
    sigma_min_gate_width: float = 0.1,
) -> InfoNoiseState:
    """Create an InfoNoiseState from an SDE instance.

    Sigma range is derived from the SDE's marginal_params at t=1e-5 and t=T.

    Args:
        sde: SDE instance with marginal_params(t) and T attribute.
        n_bins: Number of bins in log-sigma space.
        buffer_capacity: Max observations per bin FIFO buffer.
        ema_alpha: EMA smoothing coefficient for loss tracking.
        warm_up_steps: Steps to sample uniformly before switching to adaptive.
        refresh_interval: Steps between CDF rebuilds.
        sigma_min_gate_width: Gate width parameter for boundary regularization.

    Returns:
        Initialized InfoNoiseState.
    """
    _, sigma_min = sde.marginal_params(1e-5)
    _, sigma_max = sde.marginal_params(sde.T)

    # Clamp to avoid log(0)
    sigma_min = max(sigma_min, 1e-6)
    sigma_max = max(sigma_max, sigma_min + 1e-6)

    log_sigma_edges = np.linspace(np.log(sigma_min), np.log(sigma_max), n_bins + 1)
    log_sigma_centers = 0.5 * (log_sigma_edges[:-1] + log_sigma_edges[1:])
    sigma_centers = np.exp(log_sigma_centers)

    bins = [
        EntropyRateBin(
            sigma_center=float(sc),
            losses=deque(maxlen=buffer_capacity),
        )
        for sc in sigma_centers
    ]

    state = InfoNoiseState(
        bins=bins,
        n_bins=n_bins,
        buffer_capacity=buffer_capacity,
        ema_alpha=ema_alpha,
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
        warm_up_steps=warm_up_steps,
        refresh_interval=refresh_interval,
        sigma_min_gate_width=sigma_min_gate_width,
        _log_sigma_edges=log_sigma_edges,
    )

    logger.debug(
        f"InfoNoiseState created: {n_bins} bins, "
        f"sigma=[{sigma_min:.4f}, {sigma_max:.4f}], "
        f"warm_up={warm_up_steps}"
    )

    return state


def _find_bin(state: InfoNoiseState, sigma: float) -> int:
    """Find the bin index for a given sigma value.

    Uses binary search on log-sigma edges.
    """
    log_sigma = np.log(max(sigma, 1e-10))
    idx = int(np.searchsorted(state._log_sigma_edges, log_sigma, side="right")) - 1
    return max(0, min(state.n_bins - 1, idx))


def record_observation(state: InfoNoiseState, sigma: float, mse_loss: float) -> None:
    """Record an (sigma, loss) observation into the appropriate bin.

    Updates the FIFO buffer and EMA loss for the bin.

    Args:
        state: InfoNoiseState to update.
        sigma: Noise std (from marginal_params).
        mse_loss: Scalar MSE loss at this sigma.
    """
    idx = _find_bin(state, sigma)
    b = state.bins[idx]
    b.losses.append(mse_loss)
    b.count += 1

    if b.count == 1:
        b.ema_loss = mse_loss
    else:
        b.ema_loss = (1 - state.ema_alpha) * b.ema_loss + state.ema_alpha * mse_loss

    state._step_count += 1

    # Invalidate CDF on refresh interval
    if state._step_count % state.refresh_interval == 0:
        state._cdf = None


def compute_entropy_rate(state: InfoNoiseState) -> np.ndarray:
    """Compute estimated entropy rate r_hat(sigma) for each bin.

    r_hat(sigma) = gate(sigma) * mmse_hat(sigma) / sigma^3

    where gate(sigma) = sigma^n / (sigma^n + c^n) suppresses boundary artifacts
    near sigma_min, with n=3 and c=sigma_min_gate_width.

    Args:
        state: InfoNoiseState with accumulated observations.

    Returns:
        Array of shape [n_bins] with entropy rate estimates.
    """
    rates = np.zeros(state.n_bins)
    c = state.sigma_min_gate_width
    gate_n = 3  # gate exponent

    for i, b in enumerate(state.bins):
        if b.count == 0:
            continue

        sigma = b.sigma_center
        mmse_hat = b.ema_loss

        # Gated regularization: gate(sigma) = sigma^n / (sigma^n + c^n)
        gate = sigma**gate_n / (sigma**gate_n + c**gate_n)

        # Entropy rate: r_hat(sigma) = gate * mmse_hat / sigma^3
        rates[i] = gate * mmse_hat / max(sigma**3, 1e-15)

    return rates


def build_sampler_cdf(state: InfoNoiseState) -> np.ndarray:
    """Build normalized CDF for inverse-CDF sampling.

    The CDF is proportional to the entropy rate, so bins with higher
    entropy rate receive more training samples.

    Args:
        state: InfoNoiseState with accumulated observations.

    Returns:
        Normalized CDF of shape [n_bins] (values in [0, 1], last element = 1).
    """
    rates = compute_entropy_rate(state)

    # Floor: minimum sampling probability per bin to avoid starvation
    floor = rates.max() * 0.01 if rates.max() > 0 else 1.0
    rates = np.maximum(rates, floor)

    cumulative = np.cumsum(rates)
    total = cumulative[-1]
    if total < 1e-15:
        # Fallback to uniform
        return np.linspace(1.0 / state.n_bins, 1.0, state.n_bins)

    cdf = cumulative / total
    return cdf


def sample_sigma(
    state: InfoNoiseState,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample n sigma values from the adaptive distribution.

    During warm-up, returns sigmas corresponding to uniform t sampling.
    After warm-up, uses inverse-CDF sampling from the entropy rate distribution.

    Args:
        state: InfoNoiseState.
        n: Number of sigma values to sample.
        rng: NumPy random Generator for reproducibility.

    Returns:
        Array of shape [n] with sampled sigma values.
    """
    if state._step_count < state.warm_up_steps:
        # During warm-up: return None to signal caller to use uniform t
        return None

    # Build or reuse CDF
    if state._cdf is None:
        state._cdf = build_sampler_cdf(state)

    cdf = state._cdf
    u = rng.uniform(0.0, 1.0, size=n)

    # Inverse-CDF sampling: find bin for each uniform sample
    bin_indices = np.searchsorted(cdf, u, side="right")
    bin_indices = np.clip(bin_indices, 0, state.n_bins - 1)

    # Sample uniformly within each bin in log-sigma space
    log_edges = state._log_sigma_edges
    log_lo = log_edges[bin_indices]
    log_hi = log_edges[bin_indices + 1]
    log_sigma = rng.uniform(log_lo, log_hi)

    return np.exp(log_sigma)


def sigma_to_t(sde, sigma: float) -> float:
    """Convert sigma to diffusion timestep t via binary search.

    Finds t such that marginal_params(t)[1] (the std) equals sigma.
    sigma(t) = sqrt(1 - alpha_bar(t)) is monotonically increasing in t.

    Args:
        sde: SDE instance with marginal_params(t) and T attribute.
        sigma: Target noise std.

    Returns:
        Timestep t in [1e-5, T].
    """
    lo, hi = 1e-5, sde.T
    target = float(sigma)

    for _ in range(50):
        mid = 0.5 * (lo + hi)
        _, std_mid = sde.marginal_params(mid)
        if std_mid < target:
            lo = mid  # need higher t for more noise
        else:
            hi = mid
        if hi - lo < 1e-7:
            break

    return 0.5 * (lo + hi)


def sigma_to_t_batch(sde, sigmas: np.ndarray) -> np.ndarray:
    """Convert an array of sigma values to timesteps.

    Args:
        sde: SDE instance.
        sigmas: Array of sigma values.

    Returns:
        Array of timesteps.
    """
    return np.array([sigma_to_t(sde, float(s)) for s in sigmas])


def get_entropy_rate_profile(state: InfoNoiseState) -> dict:
    """Export entropy rate profile as a JSON-serializable dict.

    Args:
        state: InfoNoiseState with accumulated observations.

    Returns:
        Dict with keys: sigma_centers, entropy_rates, ema_losses, counts,
        step_count, n_bins, sigma_min, sigma_max.
    """
    rates = compute_entropy_rate(state)

    return {
        "sigma_centers": [float(b.sigma_center) for b in state.bins],
        "entropy_rates": rates.tolist(),
        "ema_losses": [float(b.ema_loss) for b in state.bins],
        "counts": [b.count for b in state.bins],
        "step_count": state._step_count,
        "n_bins": state.n_bins,
        "sigma_min": state.sigma_min,
        "sigma_max": state.sigma_max,
        "warm_up_steps": state.warm_up_steps,
    }
