# Phase 3a Review: Per-Mode Noise Shaping

**Reviewer:** Tester/Reviewer agent
**Date:** 2026-04-14
**Status:** CLEAR: Proceed to runner.

---

## 1. Plan Conformance — PASS

All 6 tasks in `plans/phase3a-plan.md` have corresponding code:

| Plan Task | Implemented | Location |
|-----------|-------------|----------|
| Step 1: `ModeImportanceWeights`, `compute_mode_importance_weights()`, `shape_noise_per_mode()` | Yes | `graph_fans/phase2/noise_shaper.py:26-189` |
| Step 2: `noise_shaping` config, `mode_weights` param, updated `_get_noise()` | Yes | `graph_fans/phase2/trainer.py:42,108,188-218` |
| Step 3: `compute_mode_energy()` | Yes | `graph_fans/phase0/spectral_profiler.py:122-133` |
| Step 4: 3-way experiment script | Yes | `scripts/test_phase3a_shaping_w1.py` (400 lines) |
| Step 5: Tests | Yes | `tests/test_phase2.py:585-727` (`TestPerModeShaping` class, 7 tests) |
| Step 6: Validate & run instructions | Yes | Script includes `--validate-only` flag (line 318) |

**Families:** The experiment script defaults to `SBM(q=0.05),SBM(q=0.1),BA(m=2),SBM(q=0.01)` — matches all 4 families from the plan (line 43).

**`--validate-only` flag:** Present at line 318, routes to `validate_only()` (lines 231-271) which prints weight profiles without training.

**3-way comparison:** `METHODS = ["uniform", "band", "mode"]` at line 44, all three tested in the per-seed loop (lines 122-168).

---

## 2. Data Leakage Audit — PASS

### Weight computation uses seed-0 training split only

- **Band weights** (lines 93-98 of experiment script): computed from `ds0["train"][:50]` where `ds0` is the seed-0 dataset.
- **Mode weights** (lines 101-106 of experiment script): computed from `ds0["train"][:50]` — same training split.
- `ds0["ref"]` is **never accessed** during weight computation. The only access to `ds["ref"]` is at line 149 for W1 evaluation (comparing generated vs reference), which is correct.

### Train/test splits

- `get_or_generate_dataset()` in `dataset.py` uses separate base seeds: `train_base_seed = seed * 10000`, `ref_base_seed = seed * 10000 + 100000` (lines 51-52). No overlap.
- Datasets are cached per `{family}_seed{seed}.npz` with separate `--dataset-dir` for Phase 3a (`results/phase3a/datasets`).

### `validate_only()` function

- Also uses only `ds0["train"]` (lines 249-259). Clean.

**Verdict:** No data leakage found.

---

## 3. Mathematical Correctness — PASS

### `compute_mode_importance_weights` (noise_shaper.py:71-105)

- Formula: `pi = mode_energies / total; weights = (pi + epsilon)^(-alpha); weights /= weights.mean()`
- Matches FANS formula `g_k = (pi_k + eps)^{-alpha}`, normalized mean=1. Correct.
- Edge case: total < 1e-10 returns uniform weights. Correct.

### `shape_noise_per_mode` (noise_shaper.py:155-189)

- `coeffs = U.T @ noise` — project to eigenbasis. Correct.
- `scale = sqrt(g_k)` as tensor, `coeffs_scaled = coeffs * scale.unsqueeze(1)` — broadcasts [n_modes] over [n_modes, n_features]. Correct.
- `shaped = U @ coeffs_scaled` — project back. Correct.
- **No empirical variance normalization** — correct per plan. Orthogonal projection of i.i.d. Gaussian preserves independence across modes, so `sqrt(g_k)` scaling gives exact variance `g_k`. With only `n_features=4` values per mode, empirical std would be noisy.

### `compute_mode_energy` (spectral_profiler.py:122-133)

- `coeffs = eigenvectors.T @ features; return (coeffs**2).sum(axis=1)` — shape [n_modes].
- Parseval's theorem verified in test (line 722-726): mode energy sum equals total signal energy. Correct.

---

## 4. Backward Compatibility — PASS

### Legacy `use_spectral_noise=True` path

In `_get_noise()` (trainer.py:196-199):
```python
effective = self.config.noise_shaping
if effective == "uniform" and self.config.use_spectral_noise:
    effective = "band"  # legacy compatibility
```

When `use_spectral_noise=True` and `noise_shaping="uniform"` (the default), effective becomes `"band"`, routing to the existing `shape_noise()` function. This preserves legacy behavior.

### Test verification

`test_backward_compat_band` (test_phase2.py:668-709) trains with both `use_spectral_noise=True` and `noise_shaping="band"` using identical seeds and asserts loss histories match within atol=1e-5. This test passes.

### Existing tests unaffected

All 78 pre-existing tests pass (85 total including 7 new). No test was modified — only new tests were added.

---

## 5. Statistical Validity — PASS

- **Bonferroni correction:** `adjusted_alpha = 0.05 / 4 = 0.0125` (script line 180). Hardcoded to 4 families as specified in the plan, ensuring conservative correction even if fewer families are tested.
- **Paired t-test:** `ttest_rel(a, b)` used for all pairwise comparisons (line 189). Correct for repeated-measures design (same seeds).
- **3 pairwise comparisons per family** (uniform-band, uniform-mode, band-mode): Bonferroni corrects only for families (4), not comparisons (3). This matches the plan specification and is defensible — the three comparisons share data and are not independent. If a stricter correction is desired (4 families x 3 comparisons = 12), that would be an analyst decision, not a code bug.

---

## 6. Test Results — PASS

```
85 passed, 5 warnings in 103.74s
```

All tests pass, including:
- 7 new `TestPerModeShaping` tests
- 78 pre-existing tests (no regressions)

Warnings are benign (torch deprecation, sklearn convergence on tiny data, scipy precision on near-identical data).

---

## Findings Summary

No CRITICAL or MAJOR findings.

### MINOR observations (informational, no action required)

1. **MINOR — Bonferroni scope:** The correction is applied per family (4) but not per pairwise comparison (3). With 3 comparisons x 4 families = 12 total tests, a stricter Bonferroni would use alpha = 0.05/12 = 0.0042. The current approach (alpha = 0.0125) follows the plan and is standard for pre-planned contrasts. The analyst should note this in the final report.

2. **MINOR — `improvement_pct` sign convention:** A positive `improvement_pct` for `(m_a, m_b)` means m_b has *lower* W1 (better). The logging correctly prints this context (lines 211-216), so it is not misleading. The summary table header says "Imp%" which is directional — the analyst should verify interpretation.

3. **MINOR — Hardcoded hyperparameters:** The script hardcodes `n_train=100, n_ref=50, n_features=4, feature_mode="community"` inside `run_family()` rather than exposing them as CLI args. This is intentional per the plan ("Same hyperparameters as 2g") to isolate the single variable, but limits flexibility for exploratory runs. Acceptable for a controlled experiment.

---

## Verdict

**CLEAR: Proceed to runner.**

All checks pass. No data leakage, correct math, full backward compatibility, all 85 tests green. The implementation faithfully follows the plan.
