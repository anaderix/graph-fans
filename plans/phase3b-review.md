# Phase 3b Review: Matched Generation Noise

**Reviewer:** Tester/Reviewer agent
**Date:** 2026-04-14
**Files reviewed:**
- `graph_fans/phase2/trainer.py` (modified)
- `scripts/test_phase3b_matched_noise.py` (new)
- `tests/test_phase2.py` (new `TestMatchedNoise` class)

---

## 1. Plan Conformance — PASS

| Requirement | Status | Notes |
|---|---|---|
| `shape_gen_noise` field in TrainConfig | PASS | Line 43: `shape_gen_noise: bool = False` |
| `generate()` shapes initial noise when enabled | PASS | Lines 326-333: resolves effective mode, applies `shape_noise` or `shape_noise_per_mode` |
| 3 conditions in experiment script | PASS | Lines 43-47: uniform, band-mismatched, band-matched |
| Tests for shaped generation start | PASS | `test_generate_shaped_noise_start` (line 737) |
| Test for backward compat (uniform unaffected) | PASS | `test_generate_uniform_unaffected` (line 783) |
| Test for default=False | PASS | `test_shape_gen_noise_default_false` (line 732) |

---

## 2. Data Leakage Audit — PASS

| Check | Status | Evidence |
|---|---|---|
| Band weights from seed-0 training split only | PASS | Script lines 89-100: `ds0["train"][:50]` used for `compute_importance_weights` |
| `ds["ref"]` never accessed for weights | PASS | `ds["ref"]` appears only at line 143 for W1 evaluation against generated samples |
| Separate base seeds for train/ref | PASS | `dataset.py` lines 51-52: `train_base_seed = seed * 10000`, `ref_base_seed = seed * 10000 + 100000` |
| Weight computation uses `ds0` (seed-0) only | PASS | Lines 89-100 use `ds0`, per-seed loop at line 107 generates fresh datasets but does not recompute weights |

No data leakage found.

---

## 3. Mathematical Correctness — PASS

| Check | Status | Notes |
|---|---|---|
| `generate()` shaping matches `_get_noise()` training shaping | PASS | Both resolve effective mode identically: `noise_shaping` config + `use_spectral_noise` legacy compat. Both call the same `shape_noise()` / `shape_noise_per_mode()` functions with the same arguments (`self.eigenvectors`, `self.band_indices`, `self.importance_weights` / `self.mode_weights`). |
| Legacy compat resolution identical in both paths | PASS | Both: `if effective == "uniform" and self.config.use_spectral_noise: effective = "band"` |
| `generate()` omits `t_knee` temporal ramp | PASS (by design) | `t_knee` is deprecated (H2 removed). At t=T, the temporal ramp would apply full shaping anyway (`phi(T) = 1`), so omitting it is correct for the generation start. |
| DDIM is deterministic (eta=0) | PASS | No noise injection during reverse steps; only initial noise matters. |

---

## 4. Backward Compatibility — PASS

| Check | Status | Evidence |
|---|---|---|
| `shape_gen_noise` defaults to `False` | PASS | TrainConfig line 43 |
| Existing tests pass without modification | PASS | All 88 tests pass (see test run below) |
| Uniform shaping + `shape_gen_noise=True` = no-op | PASS | `test_generate_uniform_unaffected` confirms identical output |

---

## 5. Test Results — PASS

```
88 passed, 5 warnings in 103.36s
```

All pre-existing tests (85) pass. All new tests (3) pass:
- `TestMatchedNoise::test_shape_gen_noise_default_false` PASSED
- `TestMatchedNoise::test_generate_shaped_noise_start` PASSED
- `TestMatchedNoise::test_generate_uniform_unaffected` PASSED

---

## Findings

### MINOR-1: No test for mode shaping in generate()

The `generate()` path supports both band and mode shaping, but only band shaping is tested in `TestMatchedNoise`. The mode path uses identical branching logic and calls the same `shape_noise_per_mode` function, so risk is low. The experiment script only uses band shaping, so this is not a blocker.

**Severity:** MINOR
**Recommendation:** Could add a mode-shaping generation test in a future phase if mode shaping becomes the primary method.

### MINOR-2: Bonferroni correction hardcoded to 4 families

`n_families_for_bonferroni = 4` (script line 173) is hardcoded regardless of how many families are actually run. This is conservative (safe) when running fewer families, but overly strict. Acceptable for research code.

**Severity:** MINOR
**Recommendation:** No change needed -- conservative correction is preferred.

### MINOR-3: Experiment script uses legacy `use_spectral_noise` instead of `noise_shaping="band"`

The CONDITIONS tuple (line 43) sets `use_spectral_noise=True/False` rather than `noise_shaping="band"/"uniform"`. Both resolve to the same effective mode via the legacy compat path, so behavior is correct. Using the new API directly would be cleaner.

**Severity:** MINOR
**Recommendation:** Consider using `noise_shaping="band"` in future scripts to avoid reliance on legacy resolution. Not a blocker.

---

## Verdict

**CLEAR: Proceed to runner.**

All 5 review categories PASS. Three MINOR findings, zero CRITICAL or MAJOR issues. Implementation matches the plan, no data leakage, shaping logic is mathematically consistent between training and generation, backward compatibility is preserved, and all 88 tests pass.
