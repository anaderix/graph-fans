# Phase 6: Real Citation Network Validation

## Context

Phases 2g-3b validated spectral noise shaping on synthetic graphs (SBM, BA) with 6-12% W1 improvement. All evidence is on synthetic features with known spectral structure. Phase 6 tests whether the effect transfers to real-world citation networks (Cora, PubMed).

Three fundamental challenges: (1) real graphs have ONE feature matrix, not 100 augmented samples; (2) Cora has n=2708 nodes but the 3L GCN fails at n=200; (3) Cora features are 1433-dim sparse binary, not 4-dim continuous.

## Sub-Phases

### Phase 6a: Scale Diagnostic on Cora Subgraphs (3 days)

**Objective:** Can the 3L GCN denoise PCA-reduced Cora features on n=50-200 subgraphs?

**Method:**
1. BFS-sample 20 connected induced subgraphs at n={50, 100, 150, 200} from Cora
2. PCA-reduce features to d={4, 16, 32} (fit PCA on full Cora, apply to subgraphs)
3. Create N=100 training samples via Gaussian augmentation: `x_aug = x_real + sigma * randn`
4. Train baseline (uniform noise, 500 epochs, 3L GCN 128h, cosine+EMA)
5. Measure std_ratio, spectral_L2, per-band energy profile

**Gate:** GCN denoises at n=100 d=16 (std_ratio < 3.0 for >=15/20 subgraphs) AND spectral energy profile is non-uniform (energy ratio >= 2x).

**Files to create:**
- `graph_fans/phase6/__init__.py`
- `graph_fans/phase6/subgraph_sampler.py` — BFS extraction, connected component validation
- `graph_fans/phase6/feature_reducer.py` — PCA/TruncatedSVD wrapper
- `graph_fans/phase6/augmentation.py` — Gaussian augmentation (spectral + dropout added in 6b)
- `scripts/test_phase6a_scale.py`
- `tests/test_phase6.py`

**Reuses:** `Trainer` (unchanged), `compute_laplacian_spectrum`, `spectral_w1_summary`, `load_citation_network`

---

### Phase 6b: Augmentation Strategy Comparison (3 days)

**Objective:** Which augmentation creates the best training distribution from a single feature matrix?

**Method:** Three strategies on the best (n, d) from 6a, 20 subgraphs:
1. **Gaussian:** `x_aug = x_real + sigma * randn` (baseline from 6a)
2. **Spectral:** Add noise in eigenbasis proportional to mode energy — `sigma_k = sigma * sqrt(E_k / E_max)` — preserves spectral profile by construction
3. **Feature dropout:** Randomly zero 10-30% of nonzero features (domain-appropriate for sparse binary Cora)

Each augmentation × {uniform, band-shaped} noise. Evaluate W1 against un-augmented real features.

**Gate:** At least one augmentation produces generated features with non-degenerate spectral profiles (not collapsed to noise/constant).

**Files to modify:** `graph_fans/phase6/augmentation.py` — add spectral and dropout strategies
**Files to create:** `scripts/test_phase6b_augmentation.py`

**Depends on:** Phase 6a gate pass.

---

### Phase 6c: Core Shaping Validation on Real Features (4 days)

**Objective:** Does band-shaped noise improve W1 on real Cora features?

**Method:**
1. Extract 50 distinct subgraphs at validated scale
2. For each: augment with best strategy from 6b, train uniform vs band-shaped, generate 50 samples
3. Spectral W1 against un-augmented real features (N_ref=1 per subgraph, noise handled by averaging across 50 subgraphs)
4. Paired t-test across subgraphs
5. If Cora positive: repeat on PubMed (n=100-200 subgraphs, TF-IDF features PCA'd to d=16)

**Gate:** Band-shaped W1 < uniform W1 with p < 0.05 on Cora. Positive direction without significance = PARTIAL-GO.

**Files to create:** `scripts/test_phase6c_validation.py`

**Depends on:** Phase 6b gate pass.

---

### Phase 6e: Spectral Augmentation on Synthetic Data — Diagnostic (0.5 day, added after 6c NO-GO)

**Objective:** Phase 6c showed band shaping has zero effect on Cora features with spectral augmentation (p=0.748, 10/21 coin flip). Does this failure transfer to *synthetic* community features when we switch from independent-sample generation (Phase 2g) to spectral augmentation from a single seed?

Three hypotheses for why 6c failed:
- **H1:** Spectral augmentation makes band shaping redundant (the augmentation already respects the spectral profile).
- **H2:** Cora's PCA-reduced features lack the extreme bimodal spectral structure that community-boundary features have by construction.
- **H3:** N_ref=1 evaluation noise masks a small shaping effect.

Phase 6e isolates H1. If band shaping still helps on synthetic features with spectral augmentation, H1 is wrong and 6c failure is Cora-specific (points to H2). If shaping disappears, spectral augmentation was silently confounding the Phase 2g result.

**Method:**
1. SBM(q=0.05) at n=50, d=4 community-boundary features (match Phase 2g exactly)
2. For each of 5 seeds: generate ONE community feature matrix, spectral-augment to 100 training samples
3. Train uniform vs band-shaped (Phase 2g config: 500 epochs, 3L GCN 128h, cosine+EMA)
4. Generate 50 samples, evaluate spectral W1 against held-out reference split (also spectral-augmented from a separate seed)
5. Paired t-test across 5 seeds
6. Compare against Phase 2g baseline (independent samples): 12.5% W1 improvement, p=0.0001

**Gate:**
- Band W1 < uniform W1 with p < 0.05 → H1 rejected, 6c failure is feature-specific → run Experiment B next (community features on Cora topology)
- No significant difference → H1 confirmed, spectral augmentation eliminates shaping effect → Phase 2g's 12% was partially confounded by augmentation method

**Files to create:** `scripts/test_phase6e_synthetic_spectral_aug.py`

**Depends on:** Nothing (uses existing infrastructure from 6a/6b + synthetic generators from Phase 2).

**Effort:** ~15-30 min GPU (5 seeds × 2 methods = 10 runs at 500 epochs).

---

### Phase 6f: Cross-Protocol 2x2x2 Evaluation (0.5 day, added after 6e NO-GO)

**Objective:** Phase 2g showed +12.5% W1 improvement from band shaping with 100 independent training samples and a 50-independent reference. Phase 6e replicated the setup but swapped BOTH the training dataset *and* the evaluation reference to spectral-augmented-from-1-sample, and the effect disappeared (-2%, p=0.629). Phase 6f disambiguates: does the training-data protocol, the evaluation-reference protocol, or both kill the shaping effect?

Four training/reference combinations, each measured with uniform and band shaping (8 cells total):

| Condition | Training | Evaluation Ref | Shaping |
|-----------|----------|----------------|---------|
| A_uni | 100 independent | 50 independent | uniform |
| A_band | 100 independent | 50 independent | band |
| B_uni | 100 independent | 50 spectral-augmented | uniform |
| B_band | 100 independent | 50 spectral-augmented | band |
| C_uni | 100 spectral-augmented | 50 independent | uniform |
| C_band | 100 spectral-augmented | 50 independent | band |
| D_uni | 100 spectral-augmented | 50 spectral-augmented | uniform |
| D_band | 100 spectral-augmented | 50 spectral-augmented | band |

A replicates Phase 2g (expected +12.5%), D replicates Phase 6e (expected ~-2%). B and C separate the two protocol changes.

**Method:**
1. SBM(q=0.05), n=50, d=4, community-boundary features (identical to Phase 2g / 6e)
2. For each of 10 seeds: generate independent train (100, seeds `s*10000..s*10000+99`), independent ref (50, seeds `s*10000+100000..s*10000+100049`), 1 spectral-augmentation seed for training (`spectral_augmentation` from single sample generated at seed `s*10000+200000`), 1 different spectral-augmentation seed for reference (seed `s*10000+300000`)
3. Compute band importance weights separately per training set (mean band energy profile from first 50 samples, no leakage from ref)
4. Train 4 models per seed: (indep_train, uniform), (indep_train, band), (specaug_train, uniform), (specaug_train, band)
5. Generate 50 samples from each model, evaluate spectral W1 against BOTH reference sets. 4 models x 2 refs = 8 W1 numbers per seed

**Key comparisons:**
- A_band vs A_uni: replicates Phase 2g — expected +12.5%
- B_band vs B_uni: indep training + specaug ref — isolates the evaluation-protocol change
- C_band vs C_uni: specaug training + indep ref — isolates the training-protocol change
- D_band vs D_uni: replicates Phase 6e — expected ~flat

**Gate criteria / interpretation:**
- If A_band << A_uni **AND** B_band ≈ B_uni: **evaluation-protocol artifact** — the shaping effect is real on matching references but disappears when evaluated against spectral-augmented references
- If A_band << A_uni **AND** C_band ≈ C_uni: **training-data artifact** — independent-sample training is required for band shaping to help
- If A_band ≈ A_uni: **Phase 2g does not replicate** — a more serious problem; the original +12.5% may have been a fluke of the 5-seed sample
- If all four comparisons flat: **no shaping effect anywhere** — Phase 2g was the fluke

**Files to create:** `scripts/test_phase6f_crossprotocol.py`, `results/phase6f/Report-Phase6f.md`
**Tests added to:** `tests/test_phase6.py`

**Depends on:** Nothing (uses existing synthetic generators, spectral augmentation, and Trainer).

**Effort:** ~2.5-3h GPU (10 seeds × 4 training runs = 40 models at 500 epochs; each evaluated against 2 references).

---

### Phase 6c-1: H2 + Power-Boosted 6c (0.5 day, added after Phase 6f GO)

**Objective:** Phase 6f overturned Phase 6e's NO-GO by going from 5 → 10 seeds. Phase 6c used ~21 Cora subgraphs with spectral augmentation and reported NO-GO (p=0.748). Two questions remain:

- **H2:** Does shaping fail on Cora because its REAL PCA-reduced features lack bimodal spectral structure? To isolate this, test shaping on SYNTHETIC community-boundary features overlaid on Cora GRAPH TOPOLOGY (not real Cora features). If shaping works here, the Cora failure is feature-specific.
- **Power:** Was Phase 6c underpowered like Phase 6e? Re-run Phase 6c exactly, but with 30 subgraphs (15 more than Phase 6c's 21) to match Phase 6f's power profile.

Phase 6c-1 runs both experiments in a single script.

**Method A (H2 — community features on Cora topology):**

1. Load Cora, BFS-sample 10 subgraphs at n=100 (same scale as Phase 6c).
2. For each subgraph, generate SYNTHETIC community-boundary features on its topology via `generate_community_boundary_features(graph, n_features=4, seed=...)`.
3. Use the Phase 2g independent-sample protocol: 100 training samples (seeds `s*10000..s*10000+99`), 50 reference samples (seeds `s*10000+100000..s*10000+100049`).
4. Compute band importance weights from the first 50 training samples only (no leakage into ref).
5. Train uniform vs band-shaped (Phase 2g config: 500 epochs, 3L GCN 128h, cosine+EMA). Paired t-test across 10 subgraphs.

**Method B (power-boosted 6c — real Cora features):**

1. Load Cora, BFS-sample 30 subgraphs at n=100 (Phase 6c used ~21).
2. PCA-reduce features to d=16 via TruncatedSVD (fit on full Cora).
3. Apply spectral augmentation (best from Phase 6b): 100 training samples, 50 reference samples per subgraph, each from a separate seed.
4. Train uniform vs band-shaped per subgraph. Evaluate spectral W1 against un-augmented real features. Paired t-test across 30 subgraphs.

**Config:** n=100, d=4 (Part A) / d=16 (Part B), 500 epochs, 3L GCN 128h, cosine+EMA. Same seeds across shaping methods per subgraph.

**Gate for Part A (H2):** Band W1 < Uniform W1 with p < 0.05 (one-sided paired t-test).

**Gate for Part B (powered 6c):** Band W1 < Uniform W1 with p < 0.05 (one-sided paired t-test).

**2x2 outcome interpretation:**

| H2 result | Power-6c result | Paper framing |
|-----------|----------------|---------------|
| GO | GO | Framing A works broadly; shaping is real across synthetic AND real features, Phase 6c was underpowered. |
| GO | NO-GO | Cora's real features are the problem — shaping works on the right (bimodal-spectrum) features. Paper frames Cora failure as a feature-property issue, not a method failure. |
| NO-GO | GO | Graph topology matters more than features — Cora's topology alone kills shaping for synthetic features, yet real Cora still shows the effect. (Unlikely/contradictory.) |
| NO-GO | NO-GO | Phase 2g effect is narrow (specific to synthetic SBM topology). Framing B (spectral augmentation as the main contribution) should be primary. |

**Files to create:** `scripts/test_phase6c1_h2_plus_power.py`, `results/phase6c1/Report-Phase6c1.md`.
**Tests added to:** `tests/test_phase6.py` (smoke tests for both pipelines).

**Depends on:** Nothing (uses existing Phase 6a/6b/6f infrastructure).

**Effort:** ~2-3h GPU (10×2 + 30×2 = 80 training runs × ~100s on L40S).

---

### Phase 6d: Feature Imputation (3 days)

**Objective:** Demonstrate practical value — mask 20% of node features, generate them, measure reconstruction quality.

**Method:**
1. Hold out 20% of nodes per subgraph
2. Train on 80% features (augmented), generate for all nodes
3. Evaluate: cosine similarity, MSE, node classification accuracy on held-out features
4. Compare: uniform vs band-shaped vs baselines (neighbor averaging, mean imputation)

**Gate:** Diffusion model beats mean imputation; spectral shaping beats uniform on >50% of subgraphs.

**Files to create:** `scripts/test_phase6d_imputation.py`

**Depends on:** Phase 6c models (runs regardless of 6c gate — negative shaping result still produces usable imputation models).

---

## Key Design Decisions

- **Start with Cora only.** Smaller (n=2708), profiled in Phase 0 (4.0x energy ratio), already in codebase. Add PubMed in 6c if Cora is positive.
- **BFS subgraph sampling.** Preserves local community structure (the source of spectral non-uniformity). n=100-200 induced subgraphs.
- **PCA at d=16 as primary.** d=4 loses too much variance on 1433-dim features. d=32 unnecessary for GCN. TruncatedSVD (no centering) for sparse binary Cora.
- **N_ref=1 evaluation.** Each subgraph provides one real feature matrix as reference. Paired comparison across 50 subgraphs averages out single-sample noise.
- **GPU time ~3-4 hours total** across all sub-phases on L40S.

## Verification

1. `uv run pytest tests/test_phase6.py -v` after each sub-phase
2. Phase 6a: sanity check plots (std_ratio distribution, spectral profiles of PCA-reduced Cora)
3. Phase 6c: summary table matching Phase 2g format (family, uniform W1, spectral W1, improvement%, p-value)
4. Each sub-phase writes results to `results/phase6{a,b,c,d,e}/`
