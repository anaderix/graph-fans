---
created: 2026-03-22
reviewer: Claude Code (implementation-critic)
scope: phase2
status: complete
---

# Phase 2 Code Review — Graph-FANS Regime A Core

**Files analyzed:** 8
- `graph_fans/phase2/noise_shaper.py`
- `graph_fans/phase2/score_network.py`
- `graph_fans/phase2/sde.py`
- `graph_fans/phase2/trainer.py`
- `graph_fans/phase2/evaluate.py`
- `graph_fans/phase2/visualize.py`
- `graph_fans/phase2/run_experiment.py`
- `tests/test_phase2.py`

**Deterministic issues found:** 4 bugs, 2 design concerns
**Plan conformance gaps:** 2

---

## Summary Verdict

| Category | Result | Notes |
|---|---|---|
| Hardcoded results | PASS | No literal metric assignments found |
| Data leakage | PASS | Train/eval feature split is clean |
| Circular references | PARTIAL PASS | JSD/HKS/MMD are structurally circular but acknowledged |
| Statistical validity | FAIL | G2 gate ignores significance; H2 gate ignores direction |
| API correctness | PASS | eval/no_grad/train states handled correctly |
| Plan conformance | FAIL | 1 of 15 planned tests missing; G2 gate logic diverges from spec |

---

## Findings

### Finding 1 — G2 Gate Ignores Bonferroni-Corrected Significance — CRITICAL

**File:** `evaluate.py:377–390`

**Code:**
```python
if improvement > 0:
    h1a_pass_count += 1          # line 377 — counts raw improvement, not significance

...
h1a_pass = h1a_pass_count >= 2   # line 390 — gate is purely directional
```

**Evidence:** The plan specifies the H1-A gate as: QBE(spectral) < QBE(uniform) in high bands on ≥2/3 families **AND** paired t-test p < 0.0167 (Bonferroni-corrected). The code computes `passes_significance` (line 375) and stores it in `h1a_details`, but `h1a_pass_count` increments at line 377 on `improvement > 0` regardless of p-value. The significance flag is decorative — it never influences the G2 gate.

**Impact:** A run where spectral noise is trivially better by noise alone (e.g., 5 random seeds with n=5 gives ~50% chance of improvement > 0 by chance) will pass the gate. The Bonferroni correction exists to prevent exactly this. With n_seeds=5, the paired t-test has very low power, so requiring significance is the correct guard.

**Repair:**
```python
# Replace line 377 with:
if passes:          # requires improvement > 0 AND p < adjusted_alpha
    h1a_pass_count += 1
```

---

### Finding 2 — H2 Gate Does Not Check Correlation Direction — MAJOR

**File:** `evaluate.py:423`

**Code:**
```python
h2_pass = p_val_h2 < 0.05
```

**Evidence:** The plan specifies "Spearman ρ > 0 between optimal t_knee\* and λ₂/λ_max, p < 0.05". The code checks only `p_val_h2 < 0.05`. Spearman's test is two-tailed: a strong negative correlation (ρ ≈ -0.9) at p = 0.04 would mark H2 as PASS, which contradicts the hypothesis direction.

**Impact:** H2 is listed as supplementary to the G2 gate (`g2_pass = h1a_pass`), so this does not affect the gate outcome directly. However the H2 `pass` field in `g2_decision.json` would be wrong in the failure case, misleading any downstream consumer.

**Repair:**
```python
h2_pass = (rho > 0) and (p_val_h2 < 0.05)
```

---

### Finding 3 — JSD/HKS/MMD Metrics Always Evaluate Graph Against Itself — MAJOR

**File:** `evaluate.py:86–95`

**Code:**
```python
ev_ref = eigenvalues  # same graph
ev_gen = eigenvalues  # same graph topology
jsd = spectral_density_jsd(ev_ref, ev_gen)  # Same graph → ~0 (expected)

hks = hks_distance(ev_ref, ev_gen)  # Same graph → ~0

d_mmd = degree_mmd(graph, graph)
c_mmd = clustering_mmd(graph, graph)
t_mmd = triangle_count_mmd(graph, graph)
```

**Evidence:** Phase 2 evaluates node *feature* generation on a *fixed* graph topology. These metrics compare graph topology to itself. They will always return exactly 0 (or numerical noise). They are stored in `Phase2Results`, serialized to CSV in `h1a_results.csv`, and displayed in `run_experiment.py` summary output.

**Impact:** Not a correctness bug — the comments acknowledge this. However, these five fields occupy space in the results dataclass, the CSV, and the log output while providing zero information. They create a misleading impression that topology is being evaluated. A reader examining the CSV will see columns for `spectral_jsd` and `hks_distance` always at ~0 and may misinterpret these as passing quality checks.

The plan acknowledges that JSD and HKS are "independent checks" — but they are not independent of QBE here since they measure graph topology, not feature quality. The plan's intent was probably to eventually compare *generated graphs*, not the training graph to itself.

**Repair options (choose one):**
1. Remove the five topology metrics from `evaluate_single` entirely in this phase.
2. Keep them but document clearly in the dataclass that they are always ~0 for fixed-topology evaluation, and exclude from any reporting tables.

---

### Finding 4 — Band Count Mismatch Causes IndexError When B > 8 — MAJOR

**File:** `noise_shaper.py:94`, `trainer.py:80`, `evaluate.py:69`

**Evidence:** Phase 0 stored band energies for B=8 bands. `_load_phase0_energies` reads these as-is. If the user runs with `--bands 16`, `partition_into_bands` produces 16 `band_indices` entries, but `ImportanceWeights.weights` has 8 elements. In `shape_noise`:

```python
scale = float(np.sqrt(weights.weights[b]))   # b up to 15, weights has 8 entries
```

This raises `IndexError` for b ≥ 8. The default is B=8 so the standard run is safe, but the CLI accepts arbitrary `--bands` without validation.

**Impact:** Silent at B < 8 (uses only first B weights — probably wrong but doesn't crash), crashes at B > 8.

**Repair:** Add a guard in `compute_importance_weights` or at the point of loading:
```python
# In _get_importance_weights or in Trainer.__init__:
if importance_weights is not None and len(importance_weights.weights) != len(band_indices):
    logger.warning(
        f"B={len(band_indices)} != weights.shape={len(importance_weights.weights)}, "
        "recomputing importance weights from raw energies with correct B"
    )
    # rebin or raise
```

Alternatively validate in `shape_noise`:
```python
assert len(band_indices) == len(weights.weights), \
    f"Band count mismatch: {len(band_indices)} vs {len(weights.weights)}"
```

---

### Finding 5 — SinusoidalTimeEmbedding Division by Zero at dim=2 — MINOR

**File:** `score_network.py:30`

**Code:**
```python
emb = math.log(10000) / (half_dim - 1)
```

**Evidence:** When `time_emb_dim=2`, `half_dim=1`, and `half_dim - 1 = 0` causes `ZeroDivisionError`. The default `time_emb_dim=32` is safe; this only triggers if a caller overrides to `time_emb_dim=2`. Tests use `hidden_dim=32` and do not exercise this path.

**Impact:** Crash-on-construction. Not triggered in any current call path.

**Repair:**
```python
if half_dim > 1:
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
else:
    emb = torch.ones(half_dim, device=device)
```

---

### Finding 6 — test_training_loss_decreases Tolerance Is Too Lenient — MINOR

**File:** `tests/test_phase2.py:209`

**Code:**
```python
assert history["loss"][-1] < history["loss"][0] * 1.5
```

**Evidence:** This allows final loss to be 50% *higher* than initial loss and still pass. A buggy trainer that returns constant or slowly increasing loss would not be caught. The plan specifies "50 epochs on tiny graph" with the intent of verifying that training works — but the test would pass even if the optimizer never converged.

**Impact:** Reduces confidence in training correctness. This is a test quality issue, not a runtime bug.

**Repair:** Tighten to `< history["loss"][0]` (must strictly decrease), or track initial vs. final over a longer window:
```python
# Loss should decrease from first 5 to last 5 epochs
early = np.mean(history["loss"][:5])
late = np.mean(history["loss"][-5:])
assert late < early, f"Loss did not decrease: {early:.4f} -> {late:.4f}"
```

---

### Finding 7 — Missing Planned Integration Test (test_h2_grid_search_runs) — MINOR

**File:** `tests/test_phase2.py`

**Evidence:** The plan (task 8, integration test 15) specifies `test_h2_grid_search_runs — t_knee grid with 2 values completes`. This test does not exist in the file. The file ends after `TestIntegration.test_uniform_vs_spectral_pipeline`.

**Impact:** The H2 experiment pipeline (`run_h2_experiment`) has no smoke test. If a regression breaks the `tknee=` method string parsing or the spectral gap computation, it would only surface at full experiment runtime (~3h).

**Repair:** Add to `TestIntegration`:
```python
def test_h2_grid_search_runs(self):
    """H2 grid search with 2 t_knee values completes without error."""
    from graph_fans.phase2.evaluate import run_h2_experiment
    config = TrainConfig(
        n_epochs=5, batch_timesteps=2, seed=0, device="cpu",
        hidden_dim=32, n_layers=2, B=4, n_gen_steps=5,
    )
    df = run_h2_experiment(
        families=["SBM(q=0.05)"],
        t_knee_values=[0.10, 0.20],
        n_seeds=2,
        n_nodes=30,
        n_features=4,
        config=config,
        B=4,
        output_dir="/tmp/test_h2",
    )
    assert len(df) == 4  # 1 family * 2 t_knee * 2 seeds
    assert "spectral_gap_ratio" in df.columns
```

---

## What Is Correct

**Data leakage: CLEAN.** Training uses seed=S, reference uses seed=S+1000. The model never receives `features_ref`. Importance weights come exclusively from Phase 0 band energies (graph topology profiling), not from any evaluation data. This is the most important leakage risk and it is handled correctly.

**Score matching target: CORRECT.** `target = -noise / std` matches the standard denoising score matching objective (Song et al. 2020). The plan notation `score(x_t, t) - (-noise/std)` is equivalent.

**Reverse SDE: CORRECT.** The sign convention with negative `dt` and `sqrt(abs(dt))` for the noise term implements Euler-Maruyama correctly.

**PyTorch eval/no_grad/train lifecycle: CORRECT.** `model.eval()` is called at line 171, `@torch.no_grad()` decorates `generate()` at line 158, and `model.train()` is restored at line 184.

**Seeding: ADEQUATE.** `torch.manual_seed` and `np.random.seed` are both set in `Trainer.train()`. `torch.manual_seed` covers both CPU and CUDA devices. The graph generators use their own seed argument separately.

**Bonferroni correction: COMPUTED BUT NOT APPLIED.** The corrected alpha (0.05 / n_families) is correctly computed per family in `h1a_details`, but as noted in Finding 1, it is not used in the gate decision.

**Module completeness: COMPLETE** (minus one test). All 7 implementation tasks from the plan are present with the correct function signatures.

---

## Priority Repair Order

1. **Finding 1** (CRITICAL) — Fix G2 gate to use `passes` not `improvement > 0`. One-line change. Without this, the gate can emit GO on statistically insignificant noise.
2. **Finding 2** (MAJOR) — Add `rho > 0` to H2 pass condition. One-line change.
3. **Finding 4** (MAJOR) — Add band count assertion in `shape_noise` before running experiments with non-default `--bands`.
4. **Finding 7** (MINOR) — Add `test_h2_grid_search_runs` before first full experiment run.
5. **Finding 3** (MAJOR) — Decide whether to keep or remove always-zero topology metrics; if kept, exclude from result tables.
6. **Finding 5** (MINOR) — Guard against `time_emb_dim=2`.
7. **Finding 6** (MINOR) — Tighten loss-decrease test assertion.
