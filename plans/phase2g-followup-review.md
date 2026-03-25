# Phase 2g Follow-up: Code Review Report

**Reviewer:** Tester/Reviewer role
**Date:** 2026-03-25
**Plan:** `/home/anaderi/projects/graph-fans/plans/phase2g-followup-plan.md`
**Test suite:** 78 tests, all PASS (100.61s)

---

## Summary

| Category | Status | Notes |
|---|---|---|
| Plan conformance | PASS with gaps | Most tasks implemented; 5 plan-specified tests absent |
| Data leakage audit | PASS | Weight computation correctly uses train split only |
| Circular reference audit | PASS | Evaluation is independent of training |
| Scientific validity | MAJOR issues | t-test direction inconsistency; p-value threshold mismatch in downstream |
| Test execution | PASS | 78/78 pass |
| Single-instance training | PASS | Trainer correctly indexes into dataset |
| Dataset cache isolation | PASS | Separate dirs per scale enforced by run_scale_study.sh |

---

## Findings

### Finding 1 — Statistical Validity: t-test argument order inconsistency — MAJOR

**File:** `graph_fans/phase2/downstream.py:322`
**Code:** `t_stat, p_val = ttest_rel(s, u)  # one-sided: spectral > uniform`

**Evidence:** `scipy.stats.ttest_rel(a, b)` computes the difference `a - b`. In `downstream.py` the call is `ttest_rel(s, u)` (spectral minus uniform), so `t_stat > 0` means spectral is better. The `significant` flag at line 345 is `p_val < 0.05` (two-sided). The comment says "one-sided" but the function returns a two-sided p-value. This is inconsistent with how `test_shaping_w1.py` (line 171) and `evaluate.py` (line 328) do it: both use `ttest_rel(u, s)` (uniform minus spectral), where `t_stat > 0` means uniform is worse (i.e., spectral is better). The argument reversal in downstream means the sign of `t_stat` is inverted relative to the other two locations. The p-value is the same magnitude for a two-sided test, but if a one-sided interpretation is intended, the correct one-sided p-value would be `p_val / 2` only when the sign is correct.

**Explanation:** With 5 seeds and a borderline result (e.g., t_stat = +1.8, p_val_twosided = 0.14), the reported p-value is 0.14 regardless of argument order. However, if the analyst interprets `t_stat` sign to decide whether spectral is better (positive `t_stat` from `ttest_rel(s, u)` → spectral better), that matches. But if `improvement_pct` is negative (spectral worse), the code at line 345 will still report `significant=True` if p_val < 0.05 without checking the direction. This is the core bug: significance is declared regardless of whether the improvement is in the right direction.

**Repair:** Add a direction check to the significance flag, consistent with `test_shaping_w1.py`:
```python
# In downstream.py, replace line 345:
"significant": bool(p_val < 0.05 and s.mean() > u.mean()),
```
Alternatively, unify argument order to `ttest_rel(u, s)` everywhere and test `improvement_pct > 0 and p_val < 0.05`.

---

### Finding 2 — Statistical Validity: downstream significance threshold inconsistent with plan — MAJOR

**File:** `graph_fans/phase2/downstream.py:345`
**Code:** `"significant": bool(p_val < 0.05),`

**Evidence:** The plan (Step 3, Success Criteria) specifies `p<0.05` for downstream. However the `analyze_downstream.py` caption (line 104) correctly states `* = significant improvement (p<0.05)`. The value 0.05 is correct per plan. The issue is that the significance flag at line 345 does not check the direction of the effect (see Finding 1). The threshold itself (0.05) matches the plan.

**Explanation:** This is a correctness bug: a result where spectral is *worse* than uniform could be flagged as significant.

**Repair:** See Finding 1 repair — add the direction check.

---

### Finding 3 — Plan Conformance: 5 plan-specified test functions are absent — MINOR

**File:** `tests/test_phase2.py`, `tests/test_downstream.py`

**Evidence:** The plan specifies the following named tests that are not present:

| Plan-specified test | Present? | Location |
|---|---|---|
| `test_run_family_smoke` | NO | Step 1 test plan |
| `test_analyze_loads_json` | NO | Step 1 test plan |
| `test_scale_dataset_generation` | NO | Step 2 test plan |
| `test_sanity_check_triggers_at_200n` | NO | Step 2 test plan |
| `test_analyze_scale_loads_multi_json` | NO | Step 2 test plan |

The downstream test `test_downstream_better_than_random` is specified in the plan but present under the name `test_accuracy_above_random_chance` — coverage exists under a different name.

The plan test `test_build_node_classification_dataset_shape` expects output shape `(200, 20)` (n_samples * n_nodes, n_nodes as spectral coords), but the actual implementation returns `(200, 4)` (n_features, not n_nodes). The test in `test_downstream.py:45` correctly checks `(200, 4)` matching the implementation. The plan's shape description `(200, 20)` is a documentation error in the plan, not a code bug.

**Explanation:** Scale tests and family-smoke tests were not implemented. These are integration-level tests; their absence means the scale machinery is untested by the test suite. The analysis scripts (`analyze_scale_results.py`, `analyze_family_results.py`) have no test coverage.

**Repair:** Add to `tests/test_phase2.py`:
```python
def test_run_family_smoke():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_family("SBM(q=0.1)", n_nodes=20, n_seeds=2, device="cpu",
                            dataset_dir=tmpdir, n_epochs=5, batch_timesteps=2)
    assert "uniform_w1s" in result and "spectral_w1s" in result
    assert "p_val" in result and "improvement_pct" in result

def test_analyze_scale_loads_multi_json():
    # Build two minimal JSON files with different n_nodes and verify load_scale_results
    ...
```

---

### Finding 4 — Plan Conformance: `--n-layers` CLI argument missing from `test_shaping_w1.py` — MINOR

**File:** `scripts/test_shaping_w1.py`

**Evidence:** The plan (Step 2, Implementation Task 1) says: "Additional change: accept `--hidden-dim` (default `128`) to support SCALE-4 retry with `256`." This is implemented. But `run_family` accepts `n_layers` as a parameter (line 48) and `run_scale_study.sh` does not pass `--n-layers`. More importantly, `main()` in `test_shaping_w1.py` parses `--hidden-dim` (line 248) but there is no `--n-layers` argparse argument. The `n_layers` parameter in `run_family` always defaults to 3 with no CLI override path.

**Explanation:** This is not a leakage issue, but the lack of `--n-layers` means the plan's SCALE-4 retry only adjusts hidden_dim and cannot test `n_layers` variation. Low severity since the plan does not specifically require `--n-layers` as a CLI argument.

**Repair:** Add to `test_shaping_w1.py`:
```python
parser.add_argument("--n-layers", type=int, default=3,
                    help="Score network GCN layers (default: 3)")
```
And pass `n_layers=args.n_layers` in the `run_family` call.

---

### Finding 5 — Plan Conformance: `run_downstream.py` missing `--n-layers` argument — MINOR

**File:** `scripts/run_downstream.py`

**Evidence:** `run_downstream_experiment` accepts `n_layers` (default 3 via the function signature in `downstream.py:209`), but `run_downstream.py` does not expose `--n-layers` as a CLI argument. The `hidden_dim` argument is exposed (line 79), but `n_layers` is hardwired to 3 in `downstream.py` and there is no way to override from the CLI.

**Explanation:** Minor omission; `n_layers=3` is the correct default and consistent with Phase 2g.

**Repair:** Add `--n-layers` argument to `run_downstream.py` parser and pass to `run_downstream_experiment`.

---

### Finding 6 — Data Leakage: Confirmed CLEAN — NOTE

**File:** `scripts/test_shaping_w1.py:85-95`, `graph_fans/phase2/downstream.py:247-254`

**Evidence:** Both scripts compute importance weights from `ds0["train"][:50]` (the first 50 samples of the seed-0 training split). The variable `ds0["ref"]` is never accessed during weight computation. Confirmed by grep: no reference to `ds0["ref"]` or `ds["ref"]` appears in the weight computation paths.

**Explanation:** No leakage. The ref split is only used as the evaluation target at lines 138 and 295–301 respectively.

---

### Finding 7 — Data Leakage: downstream classifier confirmed CLEAN — NOTE

**File:** `graph_fans/phase2/downstream.py:162-178`

**Evidence:** `evaluate_node_classification` builds `X_train, y_train` from `features_gen` (model output) and `X_test, y_test` from `features_ref` (real held-out data). `clf.fit(X_train, y_train)` is called before any access to `X_test`. The two sets are built from separate calls to `build_node_classification_dataset` with non-overlapping inputs.

**Explanation:** No leakage.

---

### Finding 8 — Dataset Cache Isolation: confirmed CLEAN — NOTE

**File:** `scripts/run_scale_study.sh:29-60`, `graph_fans/phase2/dataset.py:176`

**Evidence:** The cache key in `dataset.py` is `{safe_family}_seed{seed}.npz` — it does NOT encode `n_nodes`. `run_scale_study.sh` correctly uses separate `--dataset-dir` per scale: `datasets_n100`, `datasets_n150`, `datasets_n200`. The in-code comment in `test_shaping_w1.py` (lines 11-12) explicitly documents this risk and the mitigation.

**Explanation:** No cache collision risk given the shell script as written.

---

### Finding 9 — Single-Instance Training: confirmed CLEAN — NOTE

**File:** `graph_fans/phase2/trainer.py:220-223`

**Evidence:** The training loop at line 220 samples `sample_indices = np.random.randint(0, self.n_samples, ...)` and at line 223 uses `x_0 = self.dataset[idx]` — indexing into the full N-sample dataset. `self.n_samples` is set from the dataset shape at line 114. Dataset sizes in experiments are 100 training samples. This is a proper distributional training setup.

**Explanation:** No single-instance collapse.

---

### Finding 10 — H2 Cleanup: confirmed CLEAN — NOTE

**Files:** `graph_fans/phase2/run_experiment.py`, `graph_fans/phase2/evaluate.py`, `graph_fans/phase2/visualize.py`

**Evidence:**
- `run_experiment.py` imports only `run_h1a_experiment, compute_h1a_decision` from evaluate (line 18). No H2 symbols.
- `run_experiment.py` argparse has no `--t-knee-values` (verified by `TestCleanup::test_no_t_knee_args_in_argparse` passing).
- `evaluate.py` exposes only `compute_h1a_decision` with signature `(h1a_df, B, output_dir)` — no `h2_df` parameter (verified by `TestCleanup::test_compute_h1a_decision_signature` passing).
- `visualize.py` contains no `plot_t_knee_grid`, `plot_h2_correlation`, or `plot_g2_summary`.
- `t_knee` symbol survives only in `trainer.py:36` (marked DEPRECATED) and `noise_shaper.py` (function retained but unused by main pipeline). This matches the plan's allowance.

**Explanation:** CLEANUP-1 and CLEANUP-2 are complete.

---

### Finding 11 — Effect Size Formula: two-sided test used in all locations, no formula audit needed — NOTE

**Evidence:** All three `ttest_rel` calls use the two-sided default (no `alternative` parameter). The plan mandates Bonferroni correction at p<0.025 for the family generalization test, which is implemented in `test_shaping_w1.py:176`. The `compute_h1a_decision` in `evaluate.py:332` uses `adjusted_alpha = 0.05 / len(families)` — a dynamic Bonferroni correction that adapts to the number of families tested. For 2 families this gives alpha=0.025, matching the plan.

**Explanation:** Cohen's d / effect size is not computed anywhere — the paper will need this for the write-up. This is a write-up gap, not a code bug.

---

### Finding 12 — Convergence Warning in tests — NOTE

**Evidence:** Two test runs produce `ConvergenceWarning: lbfgs failed to converge after 1000 iterations`. This occurs in the smoke tests with tiny graphs (20 nodes, 5 epochs). The warning is expected with minimal training and does not affect correctness.

**Explanation:** Not a bug. The `max_iter=1000` in `LogisticRegression` is appropriate for real runs. For smoke tests with near-random features, convergence failure is expected.

---

## Overall Assessment

The codebase is producing genuine computation. Training indexes a real dataset of N samples, evaluation uses a held-out reference split that is never accessed during weight computation or model training, and classifier train/test splits are strictly separated. The H2 cleanup is complete and verified by the test suite.

Two issues require fixing before running experiments. In `downstream.py`, the `significant` flag does not check the direction of the effect — a result where spectral is *worse* than uniform would be flagged as significant if p<0.05. This would silently invert the scientific claim for the downstream experiment. The plan-specified integration tests for the scale study are absent, leaving the scale infrastructure untested.

**Recommended actions before runner:**
1. Fix `downstream.py:345` to add direction check (Finding 1, MAJOR).
2. Optionally add the missing scale integration tests (Finding 3, MINOR).
3. Items 4 and 5 (missing CLI args) can be deferred to the run phase if `n_layers=3` default is confirmed correct.

No finding rises to CRITICAL (blocks execution). The direction-check bug (Finding 1) is MAJOR and should be fixed before interpreting downstream results, but it does not prevent the script from running.

**Status: PROCEED WITH CAUTION — fix Finding 1 before running downstream experiments.**
