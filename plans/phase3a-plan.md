# Phase 3a: Per-Mode Noise Shaping vs Per-Band Shaping

## Context

Phase 2g showed 12.5% W1 improvement with per-band spectral noise shaping (B=8 uniform-width bands) on SBM(q=0.05). The pre-Phase 3 check showed persistence-informed band boundaries aren't useful at this scale, but also raised the question: why use bands at all? The W1 metric already evaluates per eigenmode, and per-mode importance weights have 120-168x contrast range vs 12-19x for per-band. Phase 3a tests whether per-mode noise shaping (n weights instead of 8) improves upon or at least preserves the Phase 2g result.

**Hypothesis:** Per-mode shaping provides finer spectral resolution and may amplify the improvement seen in 2g — or at minimum, reproduce it without the arbitrary band discretization.

## Plan

### Step 1: Add per-mode shaping to `noise_shaper.py`

**File:** `graph_fans/phase2/noise_shaper.py`

Add `ModeImportanceWeights` dataclass (parallel to `ImportanceWeights` but with `mode_energies: np.ndarray` of shape `[n_modes]`).

Add `compute_mode_importance_weights(mode_energies, alpha, epsilon)` — same FANS formula (`g_k = (π_k + ε)^{-α}`, normalize mean=1) but per eigenmode instead of per band.

Add `shape_noise_per_mode(noise, eigenvectors, weights)`:
1. `coeffs = U.T @ noise` — project to eigenbasis
2. Scale each mode k: `coeffs[k] *= sqrt(g_k)`
3. **No empirical variance normalization** — input is i.i.d. Gaussian, orthogonal projection preserves that, so `sqrt(g_k)` scaling gives exact variance `g_k`. Per-band code normalizes because it groups multiple modes; per-mode doesn't need this. (The existing band code normalizes `band_size * n_features` values; per-mode would only have `n_features=4` values — too few for a stable std estimate.)
4. `shaped = U @ coeffs` — project back

### Step 2: Integrate into `trainer.py`

**File:** `graph_fans/phase2/trainer.py`

Add `noise_shaping: str = "uniform"` to `TrainConfig` (values: `"uniform"`, `"band"`, `"mode"`). Keep `use_spectral_noise` as a legacy alias resolved in `_get_noise()`: if `use_spectral_noise=True` and `noise_shaping="uniform"`, treat as `"band"`.

Add `mode_weights: ModeImportanceWeights | None = None` parameter to `Trainer.__init__()`.

Update `_get_noise()`:
- `"uniform"` → return raw `torch.randn`
- `"band"` → call existing `shape_noise()`
- `"mode"` → call new `shape_noise_per_mode()`

### Step 3: Add `compute_mode_energy()` to `spectral_profiler.py`

**File:** `graph_fans/phase0/spectral_profiler.py`

Thin convenience function: `coeffs = evecs.T @ features; return (coeffs**2).sum(axis=1)` → shape `[n_modes]`. The computation exists in `spectral_wasserstein.py` but not as a standalone utility.

### Step 4: Create experiment script

**File:** `scripts/test_phase3a_shaping_w1.py` (new, don't modify Phase 2g script)

3-way comparison: uniform vs band-spectral vs mode-spectral. Structure:
- Compute both band weights and mode weights from seed-0 training split
- For each seed × method: train 3L GCN (128h, 500 epochs, cosine+EMA), generate 50 samples, evaluate spectral W1
- Families: SBM(q=0.05), SBM(q=0.1), BA(m=2) — the Phase 2g winners — plus SBM(q=0.01) as negative control (unimodal spectrum, no shaping effect in 2g)
- 5 seeds, paired t-test, Bonferroni α=0.05/4=0.0125 per family for each pairwise comparison (4 families)
- Output to `results/phase3a/`
- Include `--validate-only` flag to print weight profiles without training

### Step 5: Tests

**File:** `tests/test_phase2.py`

Add tests:
- `compute_mode_importance_weights` returns correct shape, mean=1, inverse relationship
- `shape_noise_per_mode` output shape correct, total variance reasonable
- Trainer with `noise_shaping="mode"` trains without error
- Backward compat: `noise_shaping="band"` matches old `use_spectral_noise=True`

### Step 6: Validate & run

1. Run `--validate-only` locally to confirm weight profiles match expected contrast ranges
2. Run 1 seed on CPU locally to verify training converges and sanity check passes
3. Push code to GitHub, pull on remote GPU host, run full experiment there

**Remote GPU host:** `89.169.123.173` (NVIDIA L40S)

```bash
# Local: push changes
git add -A && git commit -m "Phase 3a: per-mode noise shaping" && git push

# Remote: pull and run
ssh -o IdentitiesOnly=yes -i ~/.ssh/id_rsa anaderi@89.169.123.173
cd ~/git/graph-fans && git pull && source ~/.local/bin/env

# Validate weight profiles
uv run python scripts/test_phase3a_shaping_w1.py --validate-only

# Quick single-seed GPU smoke test
uv run python scripts/test_phase3a_shaping_w1.py --families "SBM(q=0.05)" --n-seeds 1 --device cuda --n-epochs 50

# Full GPU run (4 families × 3 methods × 5 seeds)
uv run python scripts/test_phase3a_shaping_w1.py --device cuda --n-seeds 5
```

## Key design decisions

- **Scale-only normalization for per-mode** (no empirical std): analytically correct for Gaussian input through orthogonal projection. Avoids noisy 4-sample std estimates.
- **New script, don't modify 2g script**: preserves historical reproducibility.
- **Separate `ModeImportanceWeights` dataclass**: explicit type distinction at call sites, no confusion with band-level code.
- **Same hyperparameters as 2g**: n=50, 4 features, 500 epochs, 3L GCN 128h, cosine SDE, EMA 0.999 — isolates the single variable (band vs mode shaping).

## Follow-up: Phase 3b

Phase 3b (see `plans/phase3b-plan.md`) tests matched noise — shaping the initial generation noise to match training noise. Currently there's a distributional mismatch: training corrupts features with shaped noise, but generation starts from uniform `torch.randn`. Phase 3b fixes this and reuses Phase 3a results as the mismatched baseline.

## Risk: weight contrast too high

Per-mode weights span 120-168x. Mode 0 (constant eigenvector) gets weight ~0.02, while low-energy high-frequency modes get weight ~2-3. The `sqrt(g_k)` scaling means noise coefficients range from 0.14x to 1.7x — this is actually moderate. If instability occurs, try `alpha=0.5` (reduces contrast) or clip max weight ratio at 20x.

## Files to modify

| File | Change |
|------|--------|
| `graph_fans/phase2/noise_shaper.py` | Add `ModeImportanceWeights`, `compute_mode_importance_weights()`, `shape_noise_per_mode()` |
| `graph_fans/phase2/trainer.py` | Add `noise_shaping` config field, `mode_weights` param, update `_get_noise()` |
| `graph_fans/phase0/spectral_profiler.py` | Add `compute_mode_energy()` |
| `scripts/test_phase3a_shaping_w1.py` | New: 3-way experiment script |
| `tests/test_phase2.py` | Add per-mode shaping tests |

## Verification

1. `uv run pytest tests/ -v` — all existing + new tests pass (run locally)
2. `uv run python scripts/test_phase3a_shaping_w1.py --validate-only` — weight profiles printed, contrast ranges verified (local or remote)
3. `uv run python scripts/test_phase3a_shaping_w1.py --families "SBM(q=0.05)" --n-seeds 1 --device cpu --n-epochs 50` — single quick run locally, loss decreases, sanity check passes
4. Full GPU run on `89.169.123.173`: `uv run python scripts/test_phase3a_shaping_w1.py --device cuda --n-seeds 5`
