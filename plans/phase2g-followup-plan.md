---
roadmap: /home/anaderi/projects/graph-fans/plans/phase2g-followup.md
scope: CLEANUP-1,2 + Step 1 (Family) + Step 2 (Scale) + Step 3 (Downstream) + Step 4 (Write-up)
created: 2026-03-25
status: draft
---

# Execution Plan: Phase 2g Follow-up — Generalization, Scale, and Downstream Validation

## Overview

Phase 2g established a statistically significant 12.5% W1 improvement from spectral noise shaping on SBM(q=0.05) at 50 nodes (p=0.0001). This follow-up plan validates that finding along three axes: (1) it generalizes beyond one graph family, (2) it persists or grows with graph size, and (3) it produces measurable gains on a practical downstream task. Before those experiments, dead code from the dropped H2 (t_knee) hypothesis is surgically removed to halve compute cost and reduce surface area. The plan ends with a structured write-up task.

## Prerequisites

- GPU with ≥8 GB VRAM (CUDA device), or sufficiently fast CPU for small-scale tests
- Phase 2g datasets pre-generated in `results/phase2f_small/datasets/` (used by `scripts/test_shaping_w1.py`)
- Phase 0 band energies exist at `results/phase0/band_energies.json`, or the pipeline falls back to uniform weights (acceptable)
- Python environment: `uv run python` from `/home/anaderi/projects/graph-fans/` resolves all imports
- All existing tests pass: `uv run pytest tests/test_phase2.py -q`

---

## CLEANUP-1 and CLEANUP-2: Remove H2 from the Pipeline

### Objective

Strip the t_knee grid-search (H2) from `run_experiment.py` and `evaluate.py`. H2 showed <1% W1 variation across all t_knee values and is definitively a no-op. Removing it halves compute, eliminates dead CLI flags, and simplifies the G2 decision to H1-A alone.

### Prerequisites

- Existing tests pass before this change (baseline)

### Implementation Tasks

1. **Module: `graph_fans/phase2/run_experiment.py`**
   - Purpose: Remove the H2 experiment block and associated import symbols
   - Changes:
     - Remove `run_h2_experiment` and `compute_g2_decision` from the import line (line 18)
     - Remove `plot_t_knee_grid` and `plot_h2_correlation` from the visualize import (lines 24-26)
     - In `run_phase2()`: delete the parameter `t_knee_values: list[float] | None = None` and its default assignment
     - In `run_phase2()`: delete the entire `# --- H2 Experiment ---` block (lines 128-145)
     - In `run_phase2()`: replace the `compute_g2_decision(h1a_df, h2_df, ...)` call with a simplified local decision: `g2 = {"decision": "GO" if h1a_df is not None else "NO-GO", "h1a": {}}` — or call a new `compute_h1a_decision()` helper described below
     - In `run_phase2()`: remove the two H2 plot calls (`plot_t_knee_grid`, `plot_h2_correlation`)
     - In `main()`: remove the `--t-knee-values` argument from `argparse`
     - In `main()`: remove the `t_knee = [float(x) for x in args.t_knee_values.split(",")]` line
     - In `main()`: remove `t_knee_values=t_knee` from the `run_phase2()` call

2. **Module: `graph_fans/phase2/evaluate.py`**
   - Purpose: Remove `run_h2_experiment` and simplify `compute_g2_decision` to H1-A only
   - Changes:
     - Delete the entire `run_h2_experiment()` function (lines 295-366)
     - Rename `compute_g2_decision(h1a_df, h2_df, B, output_dir)` to `compute_h1a_decision(h1a_df, B, output_dir)` — drop the `h2_df` parameter
     - Inside the renamed function: remove the entire H2 block (lines 419-451: `optimal_tknee`, `spectral_gaps`, `spearmanr`, bootstrap CI, `h2_pass`)
     - Simplify the returned `decision` dict to omit the `"h2"` key: return only `"decision"`, `"h1a"` keys
     - Remove the `from scipy.stats import spearmanr` import if it is only used by H2
     - Keep `from scipy.stats import ttest_rel` (still used by H1-A)
     - Update the `logger.info` lines in `compute_h1a_decision` to omit the H2 log statement

3. **Module: `graph_fans/phase2/visualize.py`**
   - Purpose: Remove H2-specific plot functions that will no longer be called
   - Changes:
     - Delete `plot_t_knee_grid()` (lines 117-145)
     - Delete `plot_h2_correlation()` (lines 148-180)
     - Delete `plot_g2_summary()` (lines 183-226), or simplify it to show only H1-A bars if it will be reused in Step 4

4. **Module: `tests/test_phase2.py`**
   - Purpose: Remove tests that exercise H2 code paths
   - Changes:
     - Delete `TestIntegration.test_h2_grid_search_runs` (lines 446-467)
     - Delete `TestTemporalRamp` class entirely (lines 115-158) — these test `shape_noise_with_temporal_ramp`, which is only used by H2; the function can be kept in `noise_shaper.py` but is no longer exercised in the main pipeline
     - Verify remaining tests still pass after deletion

### Data Requirements

None — this is a code-only refactor.

### Test Plan

- `test_cleanup_no_t_knee_args`: After cleanup, instantiating `argparse` via `run_experiment.main()` with `--help` must not show `--t-knee-values`
- `test_compute_h1a_decision_signature`: `compute_h1a_decision(h1a_df, B=8, output_dir=tmpdir)` accepts no `h2_df` argument and returns a dict with keys `"decision"` and `"h1a"`
- Run full test suite: `uv run pytest tests/test_phase2.py -q` — all remaining tests must pass

### Data Leakage Risks

- **Risk**: `shape_noise_with_temporal_ramp` is retained in `noise_shaper.py` but unused. If accidentally referenced, it silently applies H2 logic.
  **Mitigation**: After cleanup, `grep -r "t_knee" graph_fans/phase2/` must return zero hits in `run_experiment.py`, `evaluate.py`, and `trainer.py` (only `noise_shaper.py` and the now-dead `TrainConfig.t_knee` field are allowed to retain the symbol). Consider adding a deprecation comment to `TrainConfig.t_knee`.

### Run Configuration

- Command: `uv run pytest tests/test_phase2.py -q`
- Expected runtime: <2 minutes on CPU
- Output: clean test run, no H2-related test names in output

### Success Criteria

- Zero references to `run_h2_experiment`, `plot_t_knee_grid`, `plot_h2_correlation`, `--t-knee-values` in active code paths
- All remaining tests pass
- `run_phase2()` signature no longer has a `t_knee_values` parameter

### Analysis Plan

- None. This is a cleanup task; no plots or statistics to produce.

---

## Step 1: Family Generalization

### Objective

Test whether the 12.5% W1 improvement observed on SBM(q=0.05) generalizes to other graph families with varying spectral profiles. Specifically: SBM(q=0.1) (similar bimodal spectrum to q=0.05), BA(m=2) (adjacent bands, less bimodal), and BA(m=5) (bimodal, bands 0+2). If ≥2 of 3 new families show significant improvement (p<0.025), the effect is general to multi-scale spectral structure, not an artifact of one specific graph.

### Prerequisites

- CLEANUP-1,2 complete (H2 code removed)
- `scripts/test_shaping_w1.py` runnable from project root
- GPU available (recommended) or CPU with patience

### Implementation Tasks

1. **Module: `scripts/test_shaping_w1.py`**
   - Purpose: Extend the existing Phase 2g script to accept a configurable family list and write structured output per family
   - Current state: `FAMILIES = ["SBM(q=0.01)", "SBM(q=0.05)"]` hardcoded at line 24
   - Changes:
     - Add `argparse` to accept `--families` (comma-separated, default `"SBM(q=0.1),BA(m=2),BA(m=5)"`), `--n-nodes` (default `50`), `--n-seeds` (default `5`), `--device` (default `"cuda"`), `--output` (default `"results/diagnostics/family_generalization.json"`), `--dataset-dir` (default `"results/phase2f_small/datasets"`)
     - Key function addition: `run_family(family, n_nodes, n_seeds, device, dataset_dir) -> dict` encapsulating the inner loop (lines 33-106 of current script) and returning `{"family": ..., "uniform_w1s": [...], "spectral_w1s": [...], "t_stat": ..., "p_val": ..., "improvement_pct": ...}`
     - Move the Bonferroni-corrected significance check into `run_family` using threshold `alpha=0.025` (2-family Bonferroni over 3 new families: 0.05/2 — but roadmap specifies p<0.025 per family, so use that directly)
     - Write per-family results to JSON after each family completes (crash-safe: do not wait until all families finish)
     - Keep existing logic for importance weight computation from seed-0 training data profile (lines 40-47), dataset cache lookup via `get_or_generate_dataset`, and `spectral_w1_summary` evaluation
   - Key functions:
     - `run_family(family: str, n_nodes: int, n_seeds: int, device: str, dataset_dir: str) -> dict`: trains 5 seeds × 2 methods, returns per-family W1 stats and p-value
     - `main()`: parse args, call `run_family` for each family, aggregate and print summary table

2. **Module: `graph_fans/phase2/evaluate.py` (minor extension)**
   - Purpose: Ensure `_get_graph` handles all new families correctly
   - Verify: `_get_graph("SBM(q=0.1)", 50, seed=0)` calls `generate_sbm(n_nodes=50, p_inter=0.1, seed=0, feature_mode="smooth")` — already works via the existing parser on line 111
   - Verify: `_get_graph("BA(m=2)", 50, seed=0)` calls `generate_ba(n_nodes=50, m=2, seed=0, feature_mode="smooth")` — already works via line 114
   - No code changes needed if parsing is correct; add an integration test to confirm

3. **Module: `scripts/analyze_family_results.py`** (new, ~60 lines)
   - Purpose: Load `family_generalization.json` and produce summary table + spectral profile correlation plot
   - Key functions:
     - `load_results(path: str) -> pd.DataFrame`: reads JSON, creates one row per family with columns `family`, `uniform_w1_mean`, `spectral_w1_mean`, `improvement_pct`, `p_val`, `significant`
     - `plot_family_comparison(df: pd.DataFrame, save_path: str) -> None`: grouped bar chart (uniform vs spectral W1) per family with error bars, significance stars above bars
     - `plot_bimodality_correlation(df: pd.DataFrame, spectral_profiles: dict, save_path: str) -> None`: scatter of improvement_pct vs bimodality index (computed as max_band_energy / mean_band_energy from Phase 0 profiles or re-derived from a fresh graph at seed=0)
     - `print_summary_table(df: pd.DataFrame) -> None`: markdown table with columns Family, Uniform W1, Spectral W1, Improvement %, p-value, Significant

### Data Requirements

- Families: `SBM(q=0.1)`, `BA(m=2)`, `BA(m=5)` — all at `n_nodes=50`
- Dataset cache: `results/phase2f_small/datasets/` — reuses existing SBM(q=0.05) cache; new families auto-generated on first run via `get_or_generate_dataset`
- Per family: `n_train=100` realizations, `n_ref=50` reference samples (matching Phase 2g config)
- Seeds: 0–4 (5 seeds per family per method)
- TrainConfig: `n_epochs=500`, `batch_timesteps=32`, `hidden_dim=128`, `n_layers=3`, `conv_type="gcn"`, `sde_type="cosine"`, `use_ema=True`, `use_lr_scheduler=True` — identical to Phase 2g
- Total runs: 3 families × 5 seeds × 2 methods = 30 training runs

### Test Plan

- `test_get_graph_new_families`: `_get_graph("SBM(q=0.1)", 50, seed=0)` and `_get_graph("BA(m=2)", 50, seed=0)` and `_get_graph("BA(m=5)", 50, seed=0)` return `nx.Graph` with `n_nodes=50` nodes
- `test_run_family_smoke`: `run_family("SBM(q=0.1)", n_nodes=20, n_seeds=2, device="cpu", dataset_dir=tmpdir)` with `n_epochs=5, batch_timesteps=2` completes and returns dict with keys `uniform_w1s`, `spectral_w1s`, `p_val`, `improvement_pct`
- `test_analyze_loads_json`: `load_results("results/diagnostics/family_generalization.json")` returns a DataFrame with the expected columns after a real run

### Data Leakage Risks

- **Risk**: Importance weights for each family are computed from the same seed-0 training data used later to evaluate that seed. If seed-0 data informs the weights AND the seed-0 evaluation, the evaluation is contaminated.
  **Mitigation**: Compute importance weights from the aggregate profile of all seeds' training data (as done in the existing script at lines 40-47: weights come from seed-0 `ds0["train"][:50]`, which is the training split, not the ref split; the ref split is used for W1 evaluation). Confirm in code that `ds["ref"]` is never touched during weight computation.
- **Risk**: Dataset cache from Phase 2g (seed-0, SBM(q=0.05)) may have different `n_features` or `feature_mode` than new families generate.
  **Mitigation**: Pass explicit `n_features=4, feature_mode="community"` to `get_or_generate_dataset` for all new families. Cache filenames encode family and seed but not feature config; if a stale cache exists with different config, it will be silently loaded. Solution: validate via `ds["n_features"] == 4 and ds["feature_mode"] == "community"` at load time and regenerate if mismatch.

### Run Configuration

- Command: `uv run python scripts/test_shaping_w1.py --families SBM(q=0.1),BA(m=2),BA(m=5) --n-nodes 50 --n-seeds 5 --device cuda --output results/diagnostics/family_generalization.json --dataset-dir results/phase2f_small/datasets`
- Expected runtime: ~2 hours on GPU (30 training runs × ~4 min each)
- Output:
  - `results/diagnostics/family_generalization.json` — raw results per seed and family
  - `results/diagnostics/family_comparison.png` — bar chart
  - `results/diagnostics/family_bimodality_scatter.png` — bimodality correlation

### Success Criteria

- ≥2 of 3 new families show p<0.025 with positive W1 improvement (spectral < uniform)
- Bonferroni-adjusted threshold for declaring generalization: families that pass form a majority (≥2/3)

### Analysis Plan

- Per-family table: Family | Spectral profile description | Uniform W1 mean±std | Spectral W1 mean±std | Improvement % | p-value | Significant
- Grouped bar chart: uniform vs spectral W1 per family (4 bars total including Phase 2g SBM(q=0.05) reference)
- Bimodality scatter: x=bimodality index (ratio of top-2 band energies to total), y=improvement %, annotated with family names
- Conclusion statement: "Spectral shaping generalizes to [N] of 4 tested families, all with bimodal spectral profiles"

---

## Step 2: Scale Study

### Objective

Determine whether the W1 improvement is scale-dependent. Tests SBM(q=0.05) and BA(m=5) (the two families most likely to show the effect) at n_nodes ∈ {50, 100, 150, 200}. The 50-node result is already known; the remaining three scales are new. An increasing improvement trend with scale would be the strongest paper contribution. A hard cutoff (effect vanishes at some scale) establishes the model's capacity limit, which is also a publishable finding.

### Prerequisites

- CLEANUP-1,2 complete
- Step 1 complete (or can run in parallel if GPU resources allow separate processes)
- Phase 2g 50-node results available in `results/diagnostics/shaping_w1_test.json` for reference

### Implementation Tasks

1. **Module: `scripts/test_shaping_w1.py`** (extension of Step 1 changes)
   - Purpose: Accept `--n-nodes` as a parameter so the same script runs any scale
   - The `--n-nodes` argument added in Step 1 already covers this
   - Additional change: accept `--hidden-dim` (default `128`) to support SCALE-4 retry with `256`
   - Pass `hidden_dim` through to `TrainConfig` in `run_family`

2. **Module: `scripts/run_scale_study.sh`** (new shell script, ~30 lines)
   - Purpose: Orchestrate sequential scale runs, writing results to separate JSON files per scale
   - Runs:
     ```
     for N in 100 150 200; do
       uv run python scripts/test_shaping_w1.py \
         --families SBM(q=0.05),BA(m=5) \
         --n-nodes $N --n-seeds 5 --device cuda \
         --output results/diagnostics/scale_study_n${N}.json \
         --dataset-dir results/phase2_scale/datasets_n${N}
     done
     ```
   - If a `scale_study_n200.json` shows sanity-check failures (std_ratio > 3 for >3 seeds), auto-trigger the hidden_dim=256 retry:
     ```
     uv run python scripts/test_shaping_w1.py \
       --families SBM(q=0.05),BA(m=5) \
       --n-nodes 200 --n-seeds 5 --device cuda \
       --hidden-dim 256 \
       --output results/diagnostics/scale_study_n200_hd256.json \
       --dataset-dir results/phase2_scale/datasets_n200
     ```

3. **Module: `scripts/analyze_scale_results.py`** (new, ~80 lines)
   - Purpose: Load all scale JSON files and produce scaling curve plots and tables
   - Key functions:
     - `load_scale_results(result_dir: str) -> pd.DataFrame`: glob `scale_study_n*.json` and `shaping_w1_test.json` (n=50), build DataFrame with columns `n_nodes`, `family`, `uniform_w1_mean`, `spectral_w1_mean`, `improvement_pct`, `p_val`, `std_ratio_mean`
     - `plot_scaling_curve(df: pd.DataFrame, save_path: str) -> None`: line plots of improvement_pct vs n_nodes, one line per family, with shaded 1-std band; mark the capacity limit (first n_nodes where std_ratio_mean > 3) with a vertical dashed line
     - `plot_w1_vs_scale(df: pd.DataFrame, save_path: str) -> None`: raw W1 (both methods) vs n_nodes per family

### Data Requirements

- Families: `SBM(q=0.05)`, `BA(m=5)` — at `n_nodes ∈ {100, 150, 200}`
- New dataset dirs: `results/phase2_scale/datasets_n100/`, `..._n150/`, `..._n200/` — auto-generated on first run
- Per scale per family: `n_train=100` realizations, `n_ref=50` reference samples
- TrainConfig at n=100,150: `hidden_dim=128`, `n_layers=3` (same as 2g)
- TrainConfig at n=200 retry: `hidden_dim=256`, `n_layers=3`
- Seeds: 0–4
- Total runs: 2 families × 3 scales × 5 seeds × 2 methods = 60 training runs (+ 20 if 200-node retry needed)

### Test Plan

- `test_scale_dataset_generation`: `get_or_generate_dataset` with `n_nodes=100` graph generates tensors of shape `(100, 100, 4)` for train and `(50, 100, 4)` for ref
- `test_sanity_check_triggers_at_200n`: On a barely-trained model at 200 nodes with `hidden_dim=128`, `sanity_check()` should return `std_ratio > 3` (expected failure mode documented)
- `test_analyze_scale_loads_multi_json`: `load_scale_results` correctly parses multiple JSON files with different `n_nodes` values

### Data Leakage Risks

- **Risk**: Datasets for n=100 reuse the same dataset directory as n=50, causing the wrong feature shapes to be loaded from cache (cache key is `family_seed.npz`, not `family_n_seed.npz`).
  **Mitigation**: Use separate `--dataset-dir` per scale (as specified in `run_scale_study.sh` above). Cache filename currently does NOT encode `n_nodes`. Verify `dataset.py` `get_or_generate_dataset` key is `{safe_family}_seed{seed}.npz` — it is. Therefore separate dirs are mandatory.
- **Risk**: At larger graphs, the importance weights (computed from seed-0, n_nodes graph) may be inconsistent if graph topology changes with scale.
  **Mitigation**: Regenerate importance weights separately per scale run using `_get_graph(family, n_nodes, seed=0)` and the corresponding dataset. The `run_family` function must use the n_nodes-specific graph for weight computation.

### Run Configuration

- SCALE-1 (n=100): `uv run python scripts/test_shaping_w1.py --families SBM(q=0.05),BA(m=5) --n-nodes 100 --n-seeds 5 --device cuda --output results/diagnostics/scale_study_n100.json --dataset-dir results/phase2_scale/datasets_n100`
- SCALE-2 (n=150): same with `--n-nodes 150`, `..._n150` dirs
- SCALE-3 (n=200): same with `--n-nodes 200`, `..._n200` dirs
- SCALE-4 (n=200, hd=256): add `--hidden-dim 256`, output to `scale_study_n200_hd256.json`
- Expected runtime: ~2h per scale (GPU) → 6–8h total; SCALE-3/4 may be longer due to larger graphs
- Output per scale: `results/diagnostics/scale_study_n{N}.json`
- Analysis outputs: `results/diagnostics/scaling_curve.png`, `results/diagnostics/w1_vs_scale.png`, `results/diagnostics/scale_table.md`

### Success Criteria

- Effect persists at n=100 (p<0.025 on at least one family)
- If effect grows with n: every scale shows improvement, improvement % increases monotonically → strongest paper claim
- If effect is flat: constant improvement regardless of scale → moderately strong claim
- If effect vanishes at n=150 or n=200: publishable as capacity-limited finding; document std_ratio as the diagnostic

### Analysis Plan

- Scaling curve plot: improvement_pct vs n_nodes for each family (line + shaded std band)
- W1 absolute values table: n_nodes | SBM(q=0.05) Uniform W1 | SBM(q=0.05) Spectral W1 | BA(m=5) Uniform W1 | BA(m=5) Spectral W1
- Sanity check summary: n_nodes | mean std_ratio (uniform) | mean std_ratio (spectral) — flags capacity limit
- Decision: explicit statement "Model capacity saturates at N=X nodes with hidden_dim=128 (std_ratio > 3). Retry with hidden_dim=256: [result]."

---

## Step 3: Downstream Task Evaluation

### Objective

Determine whether the 12.5% W1 improvement in distributional fidelity translates to a measurable gain on a practical task: node classification (predict community membership). Generated feature matrices from spectral vs uniform methods are used to train a logistic regression classifier; accuracy is measured on held-out real features. A ≥2% accuracy gain demonstrates practical significance beyond the distributional metric.

### Prerequisites

- Step 1 complete (at minimum, SBM(q=0.05) and BA(m=5) results available)
- sklearn available in the environment (`uv run python -c "import sklearn"`)

### Implementation Tasks

1. **Module: `graph_fans/phase2/downstream.py`** (new, ~120 lines)
   - Purpose: Node classification evaluation using generated vs real features
   - Key functions:
     - `generate_feature_set(trainer: Trainer, n_samples: int = 1000) -> np.ndarray`: wraps `trainer.generate(n_samples=n_samples)` — returns `[n_samples, n_nodes, n_features]`
     - `build_node_classification_dataset(features_set: np.ndarray, community_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]`: for each sample, extract per-node spectral coefficients as features (via `spectral_coefficients(features, eigenvectors)` from `spectral_wasserstein`), flatten to `[n_samples * n_nodes, n_nodes]` (spectral coords), labels are `[n_samples * n_nodes]` tiled community IDs. Returns `(X, y)`.
       - Exact signature: `build_node_classification_dataset(features_set: np.ndarray, community_labels: np.ndarray, eigenvectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]`
     - `evaluate_node_classification(features_gen: np.ndarray, features_ref: np.ndarray, community_labels: np.ndarray, eigenvectors: np.ndarray, seed: int = 0) -> dict`: trains logistic regression on spectral coefficients derived from `features_gen`, evaluates on spectral coefficients derived from `features_ref`. Returns `{"train_acc": float, "test_acc": float}`.
       - Implementation: `LogisticRegression(max_iter=1000, random_state=seed)` from `sklearn.linear_model`; train on generated features, test on reference features
     - `run_downstream_experiment(family: str, n_nodes: int, n_seeds: int, device: str, dataset_dir: str, n_gen_samples: int = 1000) -> dict`: full experiment loop — for each seed, train uniform and spectral models using existing `run_single_experiment` logic (or replicate the Trainer setup from `test_shaping_w1.py`), generate `n_gen_samples` samples each, call `evaluate_node_classification`, aggregate results. Returns dict with `uniform_acc_per_seed`, `spectral_acc_per_seed`, `improvement_pct`, `p_val`.
   - Dependencies: `from graph_fans.phase2.spectral_wasserstein import spectral_coefficients`, `from sklearn.linear_model import LogisticRegression`, `from graph_fans.phase2.trainer import Trainer, TrainConfig`, `from graph_fans.phase2.evaluate import _get_graph`, `from graph_fans.phase2.dataset import get_or_generate_dataset`, `from graph_fans.phase0.spectral_profiler import compute_laplacian_spectrum, partition_into_bands`

2. **Module: `scripts/run_downstream.py`** (new, ~70 lines)
   - Purpose: CLI entry point to run downstream evaluation and save results
   - Key functions:
     - `main()`: parse `--families` (default `"SBM(q=0.05),BA(m=5)"`), `--n-nodes` (default `50`), `--n-seeds` (default `5`), `--device` (default `"cuda"`), `--n-gen-samples` (default `1000`), `--output` (default `"results/diagnostics/downstream_results.json"`), `--dataset-dir`; call `run_downstream_experiment` per family; write JSON; print markdown table
   - Dependencies: `graph_fans.phase2.downstream`, `argparse`, `json`

3. **Module: `scripts/analyze_downstream.py`** (new, ~50 lines)
   - Purpose: Load downstream JSON and produce comparison plot and table
   - Key functions:
     - `plot_accuracy_comparison(df: pd.DataFrame, save_path: str) -> None`: grouped bars showing uniform vs spectral classification accuracy per family, with std error bars and significance stars
     - `print_summary_table(df: pd.DataFrame) -> None`: markdown table with Family | Uniform Acc | Spectral Acc | Improvement % | p-value

4. **Module: `tests/test_downstream.py`** (new, ~40 lines)
   - Purpose: Unit and smoke tests for `downstream.py`

### Data Requirements

- Families: `SBM(q=0.05)` and `BA(m=5)` at n=50 (primary), optionally other families from Step 1
- Community labels: derived from `generate_sbm` block assignment (nodes 0–12 → community 0, 13–24 → community 1, 25–37 → community 2, 38–49 → community 3 for n=50, n_communities=4); or detected via `networkx.algorithms.community.louvain_communities`
- For BA graphs: use Louvain community detection (already in `multiscale_features.generate_community_boundary_features`)
- Generated features: `n_gen_samples=1000` per method per seed (using trained model)
- Reference features: existing `ds["ref"]` split (n_ref=50) for test evaluation
- Seeds: 0–4

### Test Plan

- `test_build_node_classification_dataset_shape`: with `features_set` of shape `(10, 20, 4)`, 4 communities, `eigenvectors` of shape `(20, 20)`, output `X` has shape `(200, 20)` and `y` has shape `(200,)`
- `test_evaluate_classification_returns_valid_accuracy`: accuracy values are in `[0, 1]`
- `test_downstream_better_than_random`: on SBM(q=0.05) with 4 communities, trained model's generated features should yield test accuracy >0.25 (random chance is 0.25); passes at seed 0 even with minimal training
- `test_run_downstream_smoke`: `run_downstream_experiment("SBM(q=0.05)", n_nodes=20, n_seeds=2, device="cpu", dataset_dir=tmpdir, n_gen_samples=10)` with `n_epochs=5` completes and returns dict with required keys

### Data Leakage Risks

- **Risk**: The classifier is trained on generated features and tested on real (reference) features from the same dataset split used for W1 evaluation. If the ref split is small (n_ref=50), test accuracy estimates are noisy but not leaking — the classifier never sees the ref split during training.
  **Mitigation**: Clearly separate training data (generated: `features_gen`) from test data (reference: `ds["ref"]`). Never mix the two within a single `fit`/`score` call.
- **Risk**: Community labels for BA graphs are detected by Louvain, which is stochastic. If the random seed for Louvain differs between two calls, community assignments may swap IDs, making labels inconsistent across seeds.
  **Mitigation**: Always call `louvain_communities(graph, seed=0)` with a fixed seed. Use the same community_labels array for all seeds within one family run.
- **Risk**: `n_gen_samples=1000` generated samples are all from the same trained model (same weights, different random init). If `trainer.generate()` is deterministic (DDIM is deterministic given the same initial noise), then 1000 "samples" may be identical or near-identical.
  **Mitigation**: In `Trainer.generate()`, the initial noise `x = torch.randn(...)` is called once per sample inside the loop (line 301 in `trainer.py`). Each call to `torch.randn` with no manual seed produces independent samples. Confirm this is the case before running; if not, ensure per-sample seed variation.

### Run Configuration

- Command: `uv run python scripts/run_downstream.py --families SBM(q=0.05),BA(m=5) --n-nodes 50 --n-seeds 5 --device cuda --n-gen-samples 1000 --output results/diagnostics/downstream_results.json --dataset-dir results/phase2f_small/datasets`
- Expected runtime: 2–4 hours GPU (10 training runs × ~4 min each) + negligible classifier time
- Output:
  - `results/diagnostics/downstream_results.json` — raw accuracy per seed
  - `results/diagnostics/downstream_accuracy.png` — bar chart
  - `results/diagnostics/downstream_table.md` — markdown table

### Success Criteria

- Spectral-generated features produce ≥2% higher node classification accuracy than uniform-generated features on SBM(q=0.05)
- Paired t-test over seeds: p<0.05

### Analysis Plan

- Accuracy comparison table: Family | Uniform test acc ± std | Spectral test acc ± std | Improvement % | p-value
- Bar chart: uniform vs spectral accuracy per family with std error bars
- If improvement is <2%: reframe as "W1 improvement does not translate to classification" → supports the conclusion that W1 is a more sensitive distributional metric than classification accuracy

---

## Step 4: Write-Up

### Objective

Draft a complete research paper around the Graph-FANS result. The paper's story arc: FANS works in image diffusion → does spectral noise shaping transfer to graphs? → 6 rounds of negative results → key discoveries (loss ≠ quality, QBE misses distributional effects) → positive result (12.5% W1 on bimodal-spectrum graphs) → generalization (Step 1) → scaling and downstream (Steps 2–3).

### Prerequisites

- CLEANUP-1,2 complete
- Step 1 complete (family generalization results in hand)
- Step 2 complete (scaling results in hand)
- Step 3 complete (downstream results in hand)
- All result JSONs and figures generated by Steps 1–3

### Implementation Tasks

1. **Document: `docs/paper-draft.md`** (new)
   - Purpose: Markdown draft of the full paper to be converted to LaTeX later
   - Sections:
     - Abstract (4 sentences: problem, method, key result, significance)
     - Introduction: motivation (FANS in images), research question (does it transfer to graphs?), preview of contributions
     - Related Work: graph diffusion models, spectral graph theory, score-based diffusion, FANS/frequency shaping
     - Method: SDE setup, spectral noise shaping mechanism (equations for `shape_noise`), importance weight formula g_b = (pi_bar_b + epsilon)^(-alpha), W1 evaluation metric (definition, why QBE fails)
     - Experiments Section 2a–2f: one table summarizing the negative result progression (family, metric used, finding, conclusion)
     - Experiments Section 2g: the main result — SBM(q=0.05) at 50 nodes, W1 improvement, significance test
     - Experiments Step 1: family generalization table and bimodality correlation
     - Experiments Step 2: scaling curve figure and interpretation
     - Experiments Step 3: downstream classification results
     - Discussion: conditions under which spectral shaping helps (bimodal spectrum), capacity limit (200 nodes / 3L GCN), t_knee null result
     - Conclusion: 4 contributions as bullet points
   - Content source: mine existing `results/Report-Phase2g.md` and `plans/` markdown files for tables and observations from phases 2a–2f

2. **Script: `scripts/build_reports.py`** (already exists — check if it generates paper-ready figures)
   - Purpose: Ensure all paper figures are generated in `results/paper_figures/`
   - Changes needed:
     - Add calls to `analyze_family_results.py`, `analyze_scale_results.py`, `analyze_downstream.py` outputs
     - Standardize figure style: `sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)` and 300 DPI PNG
     - Generate a combined 4-panel figure: (top-left) H1-A bar chart from 2g, (top-right) family generalization, (bottom-left) scaling curve, (bottom-right) downstream accuracy

3. **Document: `docs/contributions-summary.md`** (new, ~30 lines)
   - Purpose: Bullet-point checklist of the four paper contributions for the reviewer response
   - Contributions:
     1. Spectral W1 metric for evaluating graph feature diffusion models
     2. Empirical finding: spectral noise shaping improves generated distribution by 12.5% on bimodal-spectrum graphs
     3. Negative result (publishable): temporal ramp (t_knee) has zero effect — eliminates one degree of freedom
     4. Architectural insight: 3L GCN outperforms Transformer for iterative graph feature generation

### Data Requirements

- All result files from Steps 1–3
- Phase 2a–2f CSV/JSON summaries from `results/phase2*/` directories
- `results/Report-Phase2g.md` for the 2g result numbers

### Test Plan

- `test_paper_figures_exist`: after running `scripts/build_reports.py`, verify that `results/paper_figures/` contains at minimum: `h1a_main_result.png`, `family_generalization.png`, `scaling_curve.png`, `downstream_accuracy.png`
- Manual review: confirm all figures have axis labels, legends, and titles before submission

### Data Leakage Risks

- **Risk**: Reporting multiple p-values without Bonferroni correction inflates Type I error rate.
  **Mitigation**: In the paper, explicitly state the correction applied at each stage: Bonferroni within Step 1 (threshold p<0.025 per family for 2-family primary hypothesis), standard α=0.05 for Step 3 (single test per family).
- **Risk**: Selecting the best-performing family from Step 1 and reporting it as the primary result.
  **Mitigation**: The primary result (SBM(q=0.05) at 50 nodes) is fixed from Phase 2g and must not be changed based on Step 1 outcome. Step 1 is confirmatory, not exploratory.

### Run Configuration

- No GPU required for write-up
- Command (figure generation): `uv run python scripts/build_reports.py --output-dir results/paper_figures`
- Expected runtime: <5 minutes (plotting only)
- Output: `docs/paper-draft.md`, `docs/contributions-summary.md`, `results/paper_figures/*.png`

### Success Criteria

- Draft contains all four sections: Method, Experiments (2a–2g + Steps 1–3), Discussion, Conclusion
- All figures referenced in draft exist as files
- p-values and effect sizes in text match the JSON result files exactly

### Analysis Plan

- Cross-check every number in the paper against the JSON source files
- For each experiment table, include n_seeds, significance test type, and correction method in the caption

---

## Execution Order Summary

```
Week 1, Day 1 (1h):
  CLEANUP-1,2 → run test suite → commit

Week 1, Day 1–2 (2–3h GPU):
  Step 1: FAMILY-1 (run test_shaping_w1.py on 3 new families)

Week 1, Day 2–3 (6–8h GPU):
  Step 2: SCALE-1 (n=100) → SCALE-2 (n=150) → SCALE-3 (n=200)
          → SCALE-4 (n=200, hd=256) if SCALE-3 fails sanity check

Week 1, Day 2 (parallel with scale runs):
  Step 1 analysis: run analyze_family_results.py

Week 2, Day 1 (1.5 days):
  Step 3: implement downstream.py + run_downstream.py (TASK-1)
          → TASK-2 (run on SBM(q=0.05), BA(m=5))
          → TASK-3 (paired t-test, analysis)

Week 2, Day 2–3 (parallel with Step 3):
  Step 4: PAPER-1 (introduction + related work)
          PAPER-2 (experiments section 2a–2g)

Week 3:
  PAPER-3: incorporate Steps 1–3 results into paper
  Final review, figure polish, submit
```

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Effect vanishes at n=100 | Medium | Medium | Report scale limit as finding; focus paper on methodological contributions + 50-node result |
| Only 1 of 3 new families shows effect | Medium | Medium | Narrow but real: "bimodal spectral structure is necessary but not sufficient" |
| Downstream task shows <2% benefit | Medium | Low | Reframe: W1 is more sensitive than classification accuracy; supports W1 as evaluation metric contribution |
| n=200 fails with hd=128 AND hd=256 | Low | Low | Document capacity analysis; 200-node failure with hd=256 IS the finding (architectural limitation) |
| Dataset cache collision across scales | High | High | Separate dataset dirs per scale — enforced by `run_scale_study.sh` |
| Louvain community inconsistency | Medium | Medium | Fix `seed=0` in all Louvain calls; validate labels are stable across seeds |
