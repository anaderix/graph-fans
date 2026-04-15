# Phase 5a Code Review

**Reviewer:** Tester/Reviewer agent
**Date:** 2026-04-14
**Status: PASS -- proceed to runner.**

---

## 1. Plan Conformance

| Plan item | Status | Notes |
|-----------|--------|-------|
| Shared MLP architecture: d+3 input, 3 layers, 128 hidden | OK | `spectral_score_network.py` L32: `input_dim = n_features + 3`, L34-41: 3x (Linear+LayerNorm+SiLU) + final Linear |
| Per-mode schedule uses CosineScheduleSDE delegation | OK | `mode_schedule.py` delegates `alpha_bar`, `perturb`, `ddim_step` to `self.base_sde` |
| t_max(k) = T * (E_k/E_max)^0.5, clamped [0.1T, T] | OK | `mode_schedule.py` L56-57 |
| Batched forward pass (not Python loops over modes) | OK | Training: single MLP call at L189. Generation: batched MLP at L274, vectorized DDIM at L277-292. No per-mode Python loop in the hot path. |
| SpectralTrainConfig matches plan defaults | OK | 500 epochs, lr=1e-3, hidden_dim=128, n_layers=3, etc. |
| spatial_coherence.py metrics | OK | `node_neighbor_correlation`, `cross_mode_energy_correlation`, `spatial_coherence_summary` all present |
| Experiment script: 3-way comparison | OK | uniform_spatial, band_spatial, spectral_5a all implemented |
| Tests: ~15 tests across 4 classes | OK | 18 tests in 4 classes, all passing |

**Minor deviation:** Plan specifies `node_neighbor_correlation` uses Pearson correlation; code uses cosine similarity. Functionally equivalent for detecting spatial smoothness and arguably more robust (no zero-std edge case with Pearson). Acceptable.

---

## 2. Data Leakage Audit

| Check | Status | Evidence |
|-------|--------|----------|
| Mode energies E_k from training data only | OK | `spectral_trainer.py` L101: `self.mode_energies = (self.spectral_data ** 2).mean(axis=(0, 2))` where `spectral_data` is projected from `features_list` (training set) |
| Importance weights from seed-0 train only | OK | `test_phase5a.py` L73-83: uses `ds0["train"][:50]`, never touches `ds0["ref"]` |
| Eigendecomposition uses only topology | OK | `compute_laplacian_spectrum(graph)` takes only the graph, computes from `nx.normalized_laplacian_matrix` |
| W1 evaluation uses ref split only | OK | `test_phase5a.py` L121,159,197: `spectral_w1_summary(ds["ref"], gen_*, ...)` |
| Dataset cache directory | OK | Defaults to `results/phase5a/datasets`, separate from other phases |

**No leakage found.**

---

## 3. Scientific Validity

| Check | Status | Notes |
|-------|--------|-------|
| Paired t-test with Bonferroni correction | OK | `test_phase5a.py` L234: `bonferroni_alpha = 0.0125` (0.05/4). Uses `ttest_rel` at L244. |
| Epsilon-prediction target is noise | OK | `spectral_trainer.py` L192: `loss = ((eps_pred - noise) ** 2).mean()` |
| DDIM generation uses EMA weights | OK | `spectral_trainer.py` L243: `model = self.ema_model if self.ema_model is not None else self.model` |
| Per-mode schedule formula | OK | `mode_schedule.py` L56: `t_max_k = T * ratio ** energy_exponent` with `ratio = E_k / E_max`, clamped at L57 |
| Cosine SDE reuse (not reimplemented) | OK | `mode_schedule.py` imports `CosineScheduleSDE` from phase2 |

---

## 4. Correctness

| Check | Status | Notes |
|-------|--------|-------|
| Spectral projection: c = U^T @ x | OK | `spectral_trainer.py` L91-93: `U_T @ feat` |
| Reconstruction: x = U @ c | OK | `spectral_trainer.py` L296: `self.eigenvectors @ c_np` (shape [n_nodes, n_modes] @ [n_modes, d] = [n_nodes, d]) |
| MLP input: [c_k_t, lambda_k, t, E_k] | OK | `spectral_score_network.py` L62: `torch.cat([c_k_t, lambda_k, t, E_k], dim=-1)` giving d+3=7 dims |
| DDIM grid per mode: t_max(k) to 0 (decreasing) | OK | `mode_schedule.py` L88: `np.linspace(t_max, 1e-5, n_steps + 1)` |
| generate() reconstructs x from all modes | OK | After per-mode DDIM loop, `eigenvectors @ c_np` reconstructs spatial features |
| Forward diffusion formula | OK | `spectral_trainer.py` L181: `c_t = sqrt(ab) * c_0 + sqrt(1-ab) * noise` |
| DDIM step formula (vectorized) | OK | Lines 283,292: standard DDIM with `x_0_hat` prediction and interpolation |
| Initial noise scaling in generation | OK | Lines 260-261: scales by `sqrt(1 - alpha_bar(t_max(k)))` to match the forward process marginal |
| Dimension consistency throughout | OK | All tensors verified: [n_modes, d], [n_modes, 1] conditioning, [n_nodes, d] output |

---

## 5. Backward Compatibility

```
uv run pytest tests/ -v
144 passed, 5 warnings in 114.10s
```

All 144 tests pass (126 pre-existing + 18 new Phase 5 tests). No regressions.

---

## 6. Observations (Non-blocking)

1. **Cosine similarity vs Pearson:** `node_neighbor_correlation` uses cosine similarity instead of Pearson as described in the plan docstring/pseudocode. The actual code docstring correctly says "cosine similarity." Both detect spatial smoothness. No action needed.

2. **Generation DDIM: all modes share the same step count.** Since each mode has a different `t_max(k)`, modes with smaller `t_max` get finer-grained steps (smaller dt per step). This is actually beneficial -- low-energy modes (small t_max) get higher resolution. Good design choice.

3. **`__init__.py` is empty.** This is fine; the package is imported by explicit module paths in tests and scripts.

4. **Experiment script** uses `results/phase5a/datasets` as the default cache directory rather than reusing `results/phase4a/datasets` as mentioned in the plan. The `--dataset-dir` flag allows overriding. Not a problem -- separate cache is cleaner.

5. **Train loop samples one realization per epoch** (line 163). This is intentional per the plan and matches the spectral domain where each "sample" is the full [n_modes, d] coefficient matrix. With 500 epochs and 100 training samples, each realization is seen ~5 times on average.

---

## Summary

No CRITICAL or blocking findings. Code faithfully implements the Phase 5a plan with correct spectral projection/reconstruction, proper data separation, standard DDIM, and efficient batched computation. All existing tests continue to pass.
