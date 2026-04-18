# Phase 7-pre: Continuous-Feature Sanity Check — NO-GO

## Goal

Before running Phase 7a (multi-dataset spectral augmentation validation) and Phase 7b (downstream classification) on Elliptic Bitcoin and Stanford PPI, verify the Phase 6 pipeline (BFS subgraphs + TruncatedSVD d=16 + Gaussian augmentation + 3L GCN training) works on *continuous* features. All prior positive results (Phase 6b: +46% W1 on Cora) are on **sparse binary bag-of-words**. Continuous TF-IDF/transaction/protein features are the actual target domains. If the pipeline collapses on continuous data, the multi-dataset validation story fails before it starts.

## Method

1. Load Elliptic Bitcoin (via `torch_geometric.datasets.EllipticBitcoinDataset`) and Stanford PPI (via `torch_geometric.datasets.PPI`) — the target domains for Phase 7.
2. Extract largest connected component (Elliptic: 7,880 nodes, 9,164 edges, 165 features; PPI concatenated: 3,480 nodes, 53,377 edges, 50 features).
3. BFS-sample 10 subgraphs at n=100 per dataset.
4. Fit TruncatedSVD on full-dataset features, reduce to d=16 (matches Phase 6b operating point).
5. For 3 representative subgraphs per dataset: Gaussian augmentation to 100 training samples, train 3L GCN (128h, 500 epochs, cosine+EMA, uniform noise), evaluate std_ratio, W1, spectral L2.
6. **Gate: std_ratio < 3.0 on ≥ 2/3 sanity subgraphs per dataset.**

Two variants tested:
- **v1 (default):** global feature scale, same as Phase 6a.
- **v2 (mitigation):** per-subgraph rescaling to test whether global scale drift was the issue.

## Results — v1 (default pipeline)

### Dataset-level statistics

| Dataset | n (full) | edges | n_features | SVD var@d=16 | Energy ratio range (10 subgraphs) |
|---------|---------:|------:|-----------:|-------------:|-----------------------------------|
| Elliptic | 7,880 | 9,164 | 165 | **0.593** | 7.9 – **724** |
| PPI | 3,480 | 53,377 | 50 | **0.398** | 12.2 – 54 |
| (Cora, reference) | 2,708 | 5,429 | 1,433 | 0.152 | 3.0 – 17.0 |

**Positive finding:** SVD variance explained at d=16 is substantially better than Cora's 15.2% — Elliptic 59.3%, PPI 39.8%. The dimensionality reduction preserves much more signal on continuous features than on sparse binary bag-of-words.

### Per-subgraph training results (3 subgraphs per dataset)

**Elliptic:**

| Subgraph | Energy ratio | Final loss | std_ratio | W1 total | Gate |
|:--------:|-------------:|-----------:|----------:|---------:|:----:|
| 0 | 17.6 | 0.617 | **2.74** | 131,374 | PASS |
| 1 | 724.3 | 0.613 | **10.63** | 6,604,630 | FAIL |
| 2 | 39.7 | 0.330 | **4.30** | 169,070 | FAIL |

**Pass rate: 1/3.** Gate: **FAIL**.

**PPI:**

| Subgraph | Energy ratio | Final loss | std_ratio | W1 total | Gate |
|:--------:|-------------:|-----------:|----------:|---------:|:----:|
| 0 | 23.0 | 0.421 | **17.83** | 1,223,560 | FAIL |
| 1 | 34.6 | 0.294 | **66.75** | 22,231,001 | FAIL |
| 2 | 54.1 | 0.261 | **215.14** | 76,958,384 | FAIL |

**Pass rate: 0/3.** Gate: **FAIL**.

### Overall v1 gate: **FAIL**

Both datasets fail the ≥2/3 pass threshold. The failure pattern is consistent: the 3L GCN trains to low loss (0.26–0.62) but generates samples with 3–215× the training standard deviation. Spectral L2 distances are small (0.03–0.25) — the generated spectral profile *shape* is approximately correct, but the *magnitude* is way off. DDIM is overshooting.

## Results — v2 (per-subgraph rescale mitigation)

Attempted mitigation: rescale each subgraph's features to unit per-dim std before augmentation, then invert the rescaling at evaluation time. Rationale: Phase 6 tuned `sigma_range=(0.01, 0.5)` assuming Cora-scale features; continuous features have very different scales.

**Result:** The rescale actually made overshoot *worse* on both datasets. Elliptic's pass rate dropped from 1/3 to 0/3; PPI remained 0/3 with even larger ratios. This confirms the overshoot is not about absolute feature scale — it's about *relative* spectral asymmetry.

## Diagnosis

The failure is **DDIM generation overshoot**, not augmentation failure or GCN capacity.

1. **SVD is healthy.** Both datasets explain more variance at d=16 than Cora. Information preservation is fine.
2. **Training converges.** Loss drops to 0.26–0.62 across all 6 runs. The score network learns something.
3. **Spectral shape is preserved.** Spectral L2 distances 0.03–0.25 are small. The energy-across-modes *distribution* matches training.
4. **Magnitude blows up.** std_ratio 2.7–215 means the absolute scale of generated samples is wrong — sometimes dramatically.
5. **Correlates with extreme energy ratios.** Elliptic subgraphs with energy ratio > 300 (4/10 of them: 724, 445, 351, 303) fail hardest. Subgraph 0 with energy ratio 17.6 is the only one that passes. Moderate-asymmetry subgraphs pass; extreme-asymmetry subgraphs overshoot.

**Conjecture:** The cosine SDE schedule was tuned for Cora features that have moderate spectral asymmetry (energy ratio ~3–17). When the training distribution concentrates *most* energy into a few modes (ratio > 300), the score network correctly learns the low-energy modes but the reverse DDIM process amplifies small prediction errors on the high-energy modes. The issue is specific to the combination of:
- Extreme energy concentration in the data
- A noise schedule not adapted to that asymmetry
- Deterministic DDIM (no stochastic correction in reverse)

Per-subgraph rescaling worsens this because it equalizes the *input* scale without changing the *spectral* scale asymmetry.

## Why Cora didn't show this

Cora's sparse binary features produce energy ratios of 3–17x because bag-of-words distributes variance across many modes. Elliptic's aggregated transaction statistics concentrate variance into a few dominant directions (e.g., total transaction volume dominates). PPI's protein positional/motif features sit in between. The pipeline implicitly depends on moderate spectral asymmetry — a property Cora has and Elliptic/PPI don't.

## Gate Decision: **NO-GO**

Per the Phase 7 plan: "If both fail — stop pipeline, report failure." Phase 7a and 7b are **halted**. Infrastructure for 7a/7b is shipped and ready once the pipeline handles continuous features.

## Implications

- **Paper-readiness dropped to Tier 3.** The strongest empirical result remains Phase 6b: spectral augmentation beats Gaussian by 46% W1 on Cora. Combined with Phase 6f's SBM-scoped shaping GO, the project has a coherent single-dataset result — not a multi-dataset generalization.
- **The project has identified a new phenomenon.** Feature distributions with extreme spectral asymmetry (energy ratio > 300) break diffusion-based generation with standard DDIM. This is itself a non-trivial empirical observation that would strengthen the paper if we can characterize it.

## Recommended Path Forward

Two honest paths:

### Path A: Fix the pipeline (2-3 days of diagnostic work)

Candidate interventions to try, in order:

1. **Finer DDIM grid near t=0.** More steps in the final denoising regime where details are resolved. Cosine schedule already puts more mass here but may need more.
2. **Adaptive sigma range.** Replace `sigma_range=(0.01, 0.5)` with a per-subgraph adaptive rule based on the feature spectral profile — e.g., `sigma_max = feat_std / sqrt(max_mode_energy)`.
3. **Skip SVD entirely.** Use d=165 (Elliptic) and d=50 (PPI) native dimensionality. May eliminate the outlier amplification introduced by SVD.
4. **Larger d (32, 64).** Lower-energy modes may regularize high-energy ones, smoothing the asymmetry.
5. **Stochastic DDPM reverse process** instead of deterministic DDIM. Noise injection during reverse steps may correct prediction errors on dominant modes.
6. **Tweedie-denoising clamping.** Cap the estimated x_0 magnitude during generation.

### Path B: Reframe the paper (immediate, accept current scope)

1. **Scope the paper to Cora.** Present spectral augmentation as a novel technique validated on citation-network diffusion. Workshop-grade (GLFrontiers, GraphCon).
2. **Add theoretical analysis.** Why does spectral augmentation preserve multi-scale spectral structure? Connection to FANS (ICLR 2026) and spectrally-guided schedules (Esteves & Makadia 2026).
3. **Use Phase 7-pre as a "limitations" finding.** The paper honestly reports the extreme-asymmetry failure as identifying a boundary of applicability.

## Files

- Scripts: `scripts/test_phase7pre_sanity.py` (with `--per-subgraph-rescale` flag)
- Results: `results/phase7pre/sanity_results.json`, `results/phase7pre/sanity_results_rescaled.json`
- Loader: `graph_fans/phase7/real_dataset_loader.py`
- Tests: `tests/test_phase7.py`
- Downstream infrastructure (ready for reuse once pipeline is repaired):
  - `graph_fans/phase7/augmentation_baselines.py` — DropFeature, FeatMask, GraphMix
  - `graph_fans/phase7/downstream_classifier.py` — GCN classifier + training loop
  - `scripts/test_phase7a_multidataset.py` — shipped but not executed
