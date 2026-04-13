# Phase 3b: Matched Noise — Shaped Generation Start

## Context

In all experiments so far (Phase 2g, 3a), there is a distributional mismatch between training and generation:

- **Training forward process:** `x_t = sqrt(α_bar) * x_0 + sqrt(1-α_bar) * ε_shaped` — noise is spectrally shaped
- **Generation start:** `x_T = torch.randn(...)` — noise is uniform (spectrally flat)

At t=T, `α_bar(T) ≈ 0`, so the forward process produces `x_T ≈ ε_shaped`. But generation starts from uniform noise. The model expects shaped-noise-corrupted inputs at high t but sees flat noise instead. This mismatch propagates through all 200 DDIM reverse steps.

The 6-12% W1 improvement from Phase 2g is the residual benefit that survives despite this mismatch. The true effect of spectral shaping could be larger.

**Hypothesis:** Matching the generation start noise to the training noise distribution will amplify the spectral shaping effect.

**Prerequisite:** Phase 3a infrastructure (per-mode shaping, `noise_shaping` config field). Phase 3a results serve as the mismatched baseline.

## Plan

### Step 1: Add `shape_initial_noise` flag to `TrainConfig`

**File:** `graph_fans/phase2/trainer.py`

Add `shape_gen_noise: bool = False` to `TrainConfig`. When True, the initial x_T in `generate()` is shaped using the same noise shaper used during training.

### Step 2: Update `generate()` to optionally shape initial noise

**File:** `graph_fans/phase2/trainer.py`

```python
# In generate(), replace:
x = torch.randn(self.n_nodes, self.n_features, device=self.device)

# With:
x = torch.randn(self.n_nodes, self.n_features, device=self.device)
if self.config.shape_gen_noise and self.config.noise_shaping != "uniform":
    if self.config.noise_shaping == "band":
        x = shape_noise(x, self.eigenvectors, self.band_indices, self.importance_weights)
    elif self.config.noise_shaping == "mode":
        x = shape_noise_per_mode(x, self.eigenvectors, self.mode_weights)
```

For DDIM (deterministic, eta=0), only this initial noise matters — no noise is injected during reverse steps. So this single change aligns generation with training.

### Step 3: Create experiment script

**File:** `scripts/test_phase3b_matched_noise.py` (new)

2×2 design using the best shaping method from Phase 3a (band or mode, whichever won):

| Condition | Training noise | Generation start | Label |
|-----------|---------------|-----------------|-------|
| 1 | uniform | uniform | `uniform` (baseline) |
| 2 | shaped | uniform | `shaped-mismatched` (= Phase 3a result) |
| 3 | shaped | shaped | `shaped-matched` (new) |

Condition 2 results can be loaded from Phase 3a output (no retraining needed — just re-run generation with shaped initial noise on the same trained models). However, for clean paired comparison, retraining all 3 conditions with the same seeds is cleaner.

- Families: same as Phase 3a (SBM q=0.01, q=0.05, q=0.1, BA m=2)
- 5 seeds, paired t-test, Bonferroni correction
- n=50, 4 features, 500 epochs, 3L GCN 128h
- Output to `results/phase3b/`

### Step 4: Tests

**File:** `tests/test_phase2.py`

- `test_generate_shaped_noise_start`: verify that with `shape_gen_noise=True`, the initial noise in `generate()` has non-uniform spectral profile
- `test_generate_shaped_matches_training_distribution`: verify initial noise spectral profile matches training noise profile

### Step 5: Run

3 conditions × 4 families × 5 seeds = 60 runs. ~5h CPU or ~1.5h GPU.

**Key comparison:** shaped-matched vs shaped-mismatched (conditions 3 vs 2). If matched improves significantly over mismatched, the noise distribution alignment matters and the true shaping effect is larger than Phase 2g/3a measured.

## Expected outcomes

1. **Matched > Mismatched:** The mismatch was a real bottleneck. Report the matched result as the true effect size.
2. **Matched ≈ Mismatched:** DDIM is robust to the initial noise distribution (the iterative process corrects it). The mismatch doesn't matter in practice.
3. **Matched < Mismatched:** Unlikely, but would suggest the model learned to compensate for the mismatch and the "correction" breaks this adaptation.

## Files to modify

| File | Change |
|------|--------|
| `graph_fans/phase2/trainer.py` | Add `shape_gen_noise` to config, update `generate()` |
| `scripts/test_phase3b_matched_noise.py` | New: 2×2 experiment script |
| `tests/test_phase2.py` | Add matched-noise generation tests |

## Verification

1. `uv run pytest tests/ -v` — all tests pass
2. Single-seed smoke test: `uv run python scripts/test_phase3b_matched_noise.py --families "SBM(q=0.05)" --n-seeds 1 --device cpu --n-epochs 50`
3. Full run: `uv run python scripts/test_phase3b_matched_noise.py --device cpu --n-seeds 5`
