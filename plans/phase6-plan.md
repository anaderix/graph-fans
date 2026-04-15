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
4. Each sub-phase writes results to `results/phase6{a,b,c,d}/`
