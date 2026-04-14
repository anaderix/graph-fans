# Phase 4a + 4b Execution Plan

## Phase 4a: InfoNoise-Guided Training

### Objective
Determine whether data-adaptive timestep sampling based on the empirical conditional entropy rate improves spectral W1 for graph feature diffusion. Secondarily, characterize the "informative window" for graph features with community structure — a novel diagnostic publishable regardless of training improvement.

### Prerequisites
1. All existing tests pass
2. Phase 2g baseline reproducible: band-mismatched spectral shaping gives 6-12% W1 improvement on SBM(q=0.05) and BA(m=2) at n=50
3. Cached datasets exist for target families and seeds
4. NS-D diagnostic demonstrates gradient misallocation (54% gradient to mid-noise, 14% to underfit low-noise)

### Implementation Tasks

**Stage 1: Core InfoNoise Module**

New file: `graph_fans/phase2/info_noise.py`

Self-contained module implementing online entropy-rate estimator and adaptive sampler.

Key classes and functions:

```python
@dataclass
class EntropyRateBin:
    sigma_center: float
    losses: deque[float]  # FIFO buffer (capacity B)
    ema_loss: float
    count: int

@dataclass
class InfoNoiseState:
    bins: list[EntropyRateBin]
    n_bins: int
    buffer_capacity: int
    ema_alpha: float
    sigma_min: float
    sigma_max: float
    warm_up_steps: int
    refresh_interval: int
    _step_count: int
    _cdf: np.ndarray | None

def create_info_noise_state(sde, n_bins=20, buffer_capacity=256, ema_alpha=0.01, warm_up_steps=2000, refresh_interval=500, sigma_min_gate_width=0.1) -> InfoNoiseState
def record_observation(state, sigma, mse_loss) -> None
def compute_entropy_rate(state) -> np.ndarray  # r_hat(sigma) = mmse_hat(sigma) / sigma^3, with gated regularization
def build_sampler_cdf(state) -> np.ndarray  # normalized CDF for inverse-CDF sampling
def sample_sigma(state, n, rng) -> np.ndarray  # uniform during warm-up, adaptive after
def sigma_to_t(sde, sigma) -> float  # binary search on sigma(t) = sqrt(1 - alpha_bar(t))
def get_entropy_rate_profile(state) -> dict  # JSON-serializable export
```

Design rationale:
- Functional style with dataclass state, matching `noise_shaper.py` pattern
- Uses sigma space (not t space) because entropy rate = mmse(sigma)/sigma^3
- FIFO buffer + EMA smoothing per InfoNoise paper
- Gated regularization near sigma_min prevents boundary artifacts
- Warm-up period uses uniform sampler before switching to adaptive

**Stage 2: Trainer Integration**

Modified file: `graph_fans/phase2/trainer.py`

Add to TrainConfig:
```python
t_sampling: str = "uniform"  # "uniform", "log_snr", or "info_noise"
info_noise_n_bins: int = 20
info_noise_buffer_capacity: int = 256
info_noise_ema_alpha: float = 0.01
info_noise_warm_up_steps: int = 2000
info_noise_refresh_interval: int = 500
```

Changes to __init__: create InfoNoiseState when t_sampling == "info_noise"
Changes to train(): replace t-sampling block with 3-way dispatch (uniform/log_snr/info_noise), add record_observation after loss computation
Add get_entropy_rate_profile() method

Key: loss weighting is NOT changed. InfoNoise only changes which timesteps are sampled. Training objective stays identical.

**Stage 3: Experiment Script**

New file: `scripts/test_info_noise.py`

4-way comparison following `scripts/test_shaping_w1.py` pattern:
1. uniform — baseline uniform t-sampling
2. band — Phase 2g band-mismatched spectral shaping
3. info_noise — InfoNoise adaptive t-sampling, no spectral shaping
4. info_noise+band — InfoNoise + band-mismatched shaping

CLI: --families "SBM(q=0.05),BA(m=2)" --n-nodes 50 --n-seeds 5 --device cuda --output results/phase4a/info_noise_results.json --save-profiles

Data leakage safeguards: importance weights from seed-0 training split only, separate dataset-dir, reference split never accessed during training.

**Stage 4: Tests**

New file: `tests/test_info_noise.py` (~15 tests)

- TestEntropyRateBin: create_state, record_observation routing, FIFO eviction, EMA convergence
- TestEntropyRate: shape, higher where loss changes fast, gated boundary suppression, empty bins
- TestSampler: CDF sums to 1, warm-up is uniform, post-warmup is concentrated, sigma_to_t roundtrip
- TestTrainerIntegration: trainer with info_noise runs, exports profile, loss decreases
- TestInfoNoiseSmoke: full train-generate-evaluate pipeline on small graph (n=10, 50 epochs)

**Stage 5: Diagnostic Script**

New file: `scripts/plot_entropy_rate.py`

Plots: entropy rate profile r_hat(sigma) with informative window annotation, sampling density comparison (uniform vs log-SNR vs InfoNoise), per-bin loss comparison.

### Data Requirements
- Families: SBM(q=0.05) and BA(m=2), n=50, 4 features, community mode
- Seeds: 5 per family per method
- Dataset: 100 training + 50 reference per seed
- Total: 4 methods x 2 families x 5 seeds = 40 training runs
- Cached datasets: results/phase4a/datasets/

### Run Configuration

```bash
uv run python scripts/test_info_noise.py \
    --families "SBM(q=0.05),BA(m=2)" \
    --n-nodes 50 --n-seeds 5 --device cuda \
    --output results/phase4a/info_noise_results.json \
    --dataset-dir results/phase4a/datasets \
    --n-epochs 500 --save-profiles
```

Expected runtime: ~40 runs x ~90s = ~60 min on L40S GPU.

Output artifacts:
- results/phase4a/info_noise_results.json
- results/phase4a/entropy_rate_profiles/
- results/phase4a/info_noise_run.log

### Success Criteria

Primary gate: InfoNoise W1 improvement > 0% on at least 1/2 families.
Secondary: Entropy rate profile is non-trivial (clear informative window).

Thresholds:
- Strong: info_noise+band > 12% W1 improvement over uniform
- Moderate: info_noise+band > 6% improvement over band-only
- Marginal: any improvement > 0% with p < 0.025
- Diagnostic: entropy rate peak identified (publishable regardless)

### Analysis Plan
1. W1 comparison table: 4-method x 2-family, paired t-test
2. Entropy rate profile figure with informative window
3. Per-band W1 breakdown
4. Sampling density figure
5. Interaction analysis (super-additive?)
6. NS-D comparison: per-SNR-bin loss with InfoNoise

---

## Phase 4b: InfoGrid for DDIM

### Objective
Determine whether non-uniform DDIM step spacing (concentrated in the informative window) improves generation quality and/or efficiency vs uniform spacing.

### Prerequisites
1. Phase 4a completed: entropy rate profiles available
2. Trained models available (reuse from 4a or retrain)
3. info_noise.py module functional

### Implementation Tasks

**Stage 1: InfoGrid Module**

New file: `graph_fans/phase2/info_grid.py`

```python
def build_info_grid(entropy_rate_profile, sde, n_steps=200) -> np.ndarray
    # 1. Read r_hat(sigma), build cumulative info coordinate u(sigma)
    # 2. Space n_steps points uniformly in u-space
    # 3. Map back to sigma-space, convert to t-space
    # Returns [n_steps + 1] timesteps from T to 0

def build_uniform_grid(sde, n_steps=200) -> np.ndarray

def visualize_grids(uniform_grid, info_grid, entropy_rate_profile, save_path=None)
```

**Stage 2: Trainer Integration**

Modified: `graph_fans/phase2/trainer.py`

Add ddim_grid: str = "uniform" to TrainConfig.
Add generate_with_grid(ts, n_samples) method for external grid injection.
Modify generate() to use InfoGrid when ddim_grid == "info".

**Stage 3: Experiment Script**

New file: `scripts/test_info_grid.py`

Tests 3 axes:
- Grid type: uniform vs InfoGrid
- Step budget: 200, 100, 50 steps
- Training method: uniform vs band-shaped

CLI: --families "SBM(q=0.05),BA(m=2)" --n-seeds 5 --step-budgets "200,100,50" --profile-dir results/phase4a/entropy_rate_profiles

Total: 2 grids x 3 budgets x 2 training x 2 families x 5 seeds = 120 generation runs (only 20 training runs)

**Stage 4: Tests**

New file: `tests/test_info_grid.py` (~7 tests)

- grid length, starts at T / ends at 0, monotonically decreasing, concentrates in high-entropy region
- generate_with_grid runs, uniform grid matches default generate

### Run Configuration

```bash
uv run python scripts/test_info_grid.py \
    --families "SBM(q=0.05),BA(m=2)" \
    --n-seeds 5 --device cuda \
    --output results/phase4b/info_grid_results.json \
    --profile-dir results/phase4a/entropy_rate_profiles \
    --step-budgets "200,100,50"
```

Expected runtime: ~20 training + 120 generation runs = ~50 min on L40S.

### Success Criteria
- Quality: 200 InfoGrid W1 < 200 uniform W1 by > 3% on 1/2 families
- Efficiency: 100 InfoGrid matches 200 uniform (2x speedup)
- Strong: 50 InfoGrid matches 200 uniform (4x speedup)

### Analysis Plan
1. W1 vs step budget curve (uniform vs InfoGrid)
2. Grid visualization on entropy rate profile
3. Per-band W1 at different step budgets
4. Training method interaction
5. Step allocation table by entropy rate quartile

---

## Execution Sequence

| Day | Activity |
|-----|----------|
| 1 | Implement info_noise.py, write unit tests |
| 2 | Integrate into trainer.py, integration tests, CPU smoke test |
| 3 | Write test_info_noise.py, CPU dry run (1 seed, 50 epochs) |
| 4 | Run Phase 4a on GPU (~60 min), save profiles |
| 5 | Analyze 4a results, entropy rate plots |
| 6 | Implement info_grid.py, generate_with_grid, tests |
| 7 | Write test_info_grid.py, run Phase 4b on GPU (~50 min) |
| 8 | Analyze 4b results, comparison plots |

## Files to Create

| File | Purpose | Lines (est.) |
|------|---------|-------------|
| graph_fans/phase2/info_noise.py | Core InfoNoise entropy rate estimator | ~200 |
| graph_fans/phase2/info_grid.py | InfoGrid non-uniform DDIM construction | ~100 |
| scripts/test_info_noise.py | Phase 4a experiment script | ~250 |
| scripts/test_info_grid.py | Phase 4b experiment script | ~200 |
| scripts/plot_entropy_rate.py | Diagnostic visualization | ~120 |
| tests/test_info_noise.py | Unit + integration tests for 4a | ~200 |
| tests/test_info_grid.py | Unit tests for 4b | ~100 |

## Files to Modify

| File | Changes |
|------|---------|
| graph_fans/phase2/trainer.py | t_sampling="info_noise", ddim_grid="info", generate_with_grid(). ~40 lines. |
