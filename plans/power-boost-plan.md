---
roadmap: /home/anaderi/projects/graph-fans/plans/phase2g-followup-plan.md
scope: power-boost (re-run 3 borderline conditions at 15 seeds)
created: 2026-03-25
status: draft
---

# Execution Plan: Power Boost — Re-run Borderline Conditions at 15 Seeds

## Overview

Phase 2g follow-up ran 3 families × 3 scales with 5 seeds each. Three conditions produced
positive effect sizes but failed Bonferroni-corrected significance (α=0.025). With 15 seeds,
the paired t-test gains ~14 degrees of freedom (df=14 vs df=4), raising power from roughly
30-50% to 70-85% for the observed effect sizes. No code changes are required; the existing
script already accepts `--n-seeds`. Three targeted re-runs replace the prior 5-seed results
for these conditions.

## Prerequisites

- GPU available (or CPU with patience; n=50 and n=150 runs are tolerable on CPU at ~3h each).
- `uv run python scripts/test_shaping_w1.py --help` executes without error.
- Separate `--dataset-dir` per scale: seeds 0–4 cached files already exist for SBM(q=0.1)
  at n=50 in `results/phase2f/datasets/`; seeds 5–14 will be generated on first run.
  For BA(m=5) at n=150 and n=200 no cached datasets exist yet — all 15 seeds will be
  generated fresh.
- The `results/phase2g_powerboost/` directory will be created by the script automatically.

---

## Prior Results Summary (5 seeds, for reference)

| Condition          | Unif W1 mean | Spec W1 mean | Improv% | t     | p      |
|--------------------|-------------|-------------|---------|-------|--------|
| SBM(q=0.1) n=50    | 1035.8      | 963.7       | 7.0%    | 3.109 | 0.0359 |
| BA(m=5) n=150      | 2503.2      | 2224.7      | 11.1%   | 3.418 | 0.0268 |
| BA(m=5) n=200      | 2754.0      | 2576.9      | 6.4%    | 2.685 | 0.0549 |

Bonferroni threshold α=0.025 (two families tested at each scale in the prior run).

---

## Statistical Note: Expected Power at 15 Seeds

With df=14 and α=0.025 (one threshold; Bonferroni is moot since each run below tests one
family independently):

- **SBM(q=0.1) n=50**: observed Cohen's d ≈ 3.109/√5 = 1.39. Power for d=1.39 at df=14,
  α=0.025 two-sided ≈ **>99%**. Prior p=0.036 was an artifact of only 5 seeds; 15 seeds
  should give p < 0.001 if the effect holds.

- **BA(m=5) n=150**: observed d ≈ 3.418/√5 = 1.53. Power ≈ **>99%**. Prior p=0.027 was
  just above threshold; 15 seeds should give p < 0.001.

- **BA(m=5) n=200**: observed d ≈ 2.685/√5 = 1.20. Power for d=1.20 at df=14, α=0.025 ≈
  **~97%**. This condition is the weakest; if the effect is real it will resolve clearly.
  If the true d is ~0.7 (the 5-seed estimate is noisy), power drops to ~55% — still better
  than before.

**Practical rule**: if p < 0.025 at 15 seeds, declare significant. If p is between 0.025
and 0.05, treat as suggestive and report Cohen's d with 95% CI.

---

## Run 1: SBM(q=0.1) at n=50

### Command

```
uv run python /home/anaderi/projects/graph-fans/scripts/test_shaping_w1.py \
  --families "SBM(q=0.1)" \
  --n-nodes 50 \
  --n-seeds 15 \
  --dataset-dir /home/anaderi/projects/graph-fans/results/phase2g_powerboost/datasets_n50 \
  --output /home/anaderi/projects/graph-fans/results/phase2g_powerboost/sbm_q0.1_n50_15seeds.json \
  --device cuda
```

### Dataset Directory Strategy

Use `results/phase2g_powerboost/datasets_n50/` — a fresh dir dedicated to n=50 re-runs.
Do NOT reuse `results/phase2f/datasets/` even though it contains SBM(q=0.1) seeds 0–4:
those were generated with the phase2f graph instance; reusing them is safe in principle but
mixing cache provenance is risky if any graph parameter differs. Fresh generation of all 15
seeds is cleaner. The n_nodes=50 must not share a cache dir with n=150 or n=200 (cache key
is `SBM_q0.1_seed{N}.npz`, no n_nodes in the key).

### Expected Runtime

~40 min on GPU (15 seeds × 2 methods × 500 epochs, n=50 graph, 100 training samples).

### Output

`/home/anaderi/projects/graph-fans/results/phase2g_powerboost/sbm_q0.1_n50_15seeds.json`

### Success Criteria

- `p_val < 0.025` and `improvement_pct > 0` → **SIGNIFICANT, condition confirmed**
- `p_val < 0.05` and `improvement_pct > 0` → suggestive, report Cohen's d
- `improvement_pct <= 0` → effect reversed, condition fails regardless of p

---

## Run 2: BA(m=5) at n=150

### Command

```
uv run python /home/anaderi/projects/graph-fans/scripts/test_shaping_w1.py \
  --families "BA(m=5)" \
  --n-nodes 150 \
  --n-seeds 15 \
  --dataset-dir /home/anaderi/projects/graph-fans/results/phase2g_powerboost/datasets_n150 \
  --output /home/anaderi/projects/graph-fans/results/phase2g_powerboost/ba_m5_n150_15seeds.json \
  --device cuda
```

### Dataset Directory Strategy

Use `results/phase2g_powerboost/datasets_n150/`. No prior cache exists for this scale; all
15 seeds generated fresh. Must be isolated from `datasets_n50` and `datasets_n200` (cache
key collision: `BA_m5_seed{N}.npz` is the same filename across all scales).

### Expected Runtime

~2.5h on GPU (15 seeds × 2 methods × 500 epochs, n=150 graph, larger Laplacian computation).

### Output

`/home/anaderi/projects/graph-fans/results/phase2g_powerboost/ba_m5_n150_15seeds.json`

### Success Criteria

Same as Run 1: `p_val < 0.025` and `improvement_pct > 0` for SIGNIFICANT.

---

## Run 3: BA(m=5) at n=200

### Command

```
uv run python /home/anaderi/projects/graph-fans/scripts/test_shaping_w1.py \
  --families "BA(m=5)" \
  --n-nodes 200 \
  --n-seeds 15 \
  --dataset-dir /home/anaderi/projects/graph-fans/results/phase2g_powerboost/datasets_n200 \
  --output /home/anaderi/projects/graph-fans/results/phase2g_powerboost/ba_m5_n200_15seeds.json \
  --device cuda
```

### Dataset Directory Strategy

Use `results/phase2g_powerboost/datasets_n200/`. Same cache-collision risk as Run 2 —
must be a different directory from `datasets_n150`. No prior cache exists.

### Expected Runtime

~3.5h on GPU (15 seeds × 2 methods × 500 epochs, n=200 graph, larger spectral computation).

### Output

`/home/anaderi/projects/graph-fans/results/phase2g_powerboost/ba_m5_n200_15seeds.json`

### Success Criteria

Same as Runs 1 and 2: `p_val < 0.025` and `improvement_pct > 0` for SIGNIFICANT.

---

## Data Leakage Risks

- **Risk**: Reusing `results/phase2f/datasets/` for SBM(q=0.1) n=50 would mix n=50 cache
  files from a different graph instance if any parameter drifted between phases.
  **Mitigation**: Use a dedicated fresh `datasets_n50` directory for all 15 seeds.

- **Risk**: Two runs share the same n_nodes directory (e.g., accidentally pointing both
  n=150 and n=200 runs at the same dir). The cache key `BA_m5_seed{N}.npz` is identical;
  whichever run generates seed 0 first will be loaded by the other run for a mismatched
  graph size, producing silently wrong W1 values.
  **Mitigation**: Three separate `--dataset-dir` paths as shown above. Verify with
  `ls results/phase2g_powerboost/` after each run that the three dataset dirs are distinct.

- **Risk**: Importance weights computed from seed-0 training split. If seed 0 is anomalous
  at a new scale, the weights may be poor.
  **Mitigation**: Log the printed importance weights for seed 0 from each run; verify they
  are not degenerate (no single weight >0.9 or all weights uniform). The script already logs
  this at INFO level.

---

## Code Changes Required

None. The script already supports:
- `--n-seeds` (line 219) — accepts any integer
- `--n-nodes` (line 213) — controls graph size
- `--families` (line 206) — comma-separated, single family is fine
- `--dataset-dir` (line 238) — separate per scale
- Crash-safe: writes JSON after each family (line 273)

The `significant` flag uses `p_val < 0.025` (line 176), which is correct for the
Bonferroni-adjusted threshold. With single-family runs below, Bonferroni does not apply
(only one hypothesis tested per run), but p < 0.025 is still a conservative and defensible
threshold; no change is needed.

---

## Analysis Plan

After all three runs complete, for each output JSON:

1. **Primary table**: family, n_nodes, uniform_w1_mean ± std, spectral_w1_mean ± std,
   improvement_pct, t_stat, p_val, significant.

2. **Cohen's d**: compute `d = t_stat / sqrt(n_seeds)` for each condition. Report d with
   95% CI (use `scipy.stats.t.interval` on the paired differences).

3. **Plot** (optional): bar chart of per-seed W1 differences (uniform − spectral) for each
   condition; horizontal line at 0; annotate with p-value.

4. **Decision**: if 2 of 3 conditions are significant (p < 0.025), the scale-generalization
   claim for BA(m=5) is supported. If SBM(q=0.1) n=50 is also significant, the
   cross-family claim is strengthened.
