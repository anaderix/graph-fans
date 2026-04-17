# Phase 7a: Multi-Dataset Validation of Spectral Augmentation

## Context

Phase 6c-1 definitively closed the band-shaping-on-real-graphs story: shaping fails on Cora topology for both synthetic and real features (Part B significant negative at p=0.004, two-sided). Framing A (band shaping as primary) is now scope-bounded to synthetic block-structured graphs.

Framing B (spectral augmentation as the practical contribution) rests on a single data point: Phase 6b showed +46% W1 improvement for spectral vs gaussian augmentation on Cora. For a paper, this needs cross-domain validation to establish generality.

**Primary research question:** Does spectral augmentation outperform gaussian augmentation on diffusion-based feature generation across multiple real citation/purchase/coauthor networks?

**Secondary:** Does the effect size correlate with dataset properties (feature dimensionality, spectral energy distribution, graph density, community clarity)?

## Objectives

1. **Replicate Phase 6b on Cora at higher power** (n=30 subgraphs vs 20) to confirm the +46% baseline is robust.
2. **Validate on PubMed** (continuous TF-IDF features, n=19717) — tests whether effect transfers from binary bag-of-words to continuous TF-IDF.
3. **Validate on Amazon Photo** (product features, n=7650) — tests non-citation domain.
4. **Validate on Coauthor CS** (keyword features, n=18333) — tests denser graph with clearer community structure.
5. **Cross-dataset analysis**: regress improvement size on dataset properties.

## Datasets

| Dataset | Nodes | Features | Type | PyG loader | Already supported |
|---------|-------|----------|------|------------|-------------------|
| Cora | 2,708 | 1,433 | binary bag-of-words | Planetoid | YES |
| PubMed | 19,717 | 500 | TF-IDF continuous | Planetoid | YES |
| Amazon Photo | 7,650 | 745 | continuous | Amazon | needs adapter |
| Coauthor CS | 18,333 | 6,805 | keyword | Coauthor | needs adapter |

## Method

### Shared protocol (matches Phase 6b)

For each dataset:
1. Load full graph + features via torch_geometric.
2. BFS-sample 30 connected subgraphs at n=100 (primary) and optionally n=200 for scaling check.
3. TruncatedSVD fit on FULL feature matrix to d=16 dimensions. Fit once per dataset.
4. For each subgraph × augmentation strategy × noise method (uniform only for this phase):
   a. Extract + reduce subgraph features to 16D.
   b. Create 100 augmented training samples + 50 reference samples (separate seeds).
   c. Train 3L GCN (128h, 500 epochs, cosine+EMA).
   d. Generate 50 samples, compute spectral W1 vs un-augmented real features.
5. Paired t-test across 30 subgraphs per dataset per augmentation pair.

### Conditions per subgraph (per dataset, 3 models × 30 subgraphs = 90 training runs per dataset)

- `gaussian`: baseline from Phase 6a
- `spectral`: the technique being validated (+46% on Cora from 6b)
- `dropout`: domain-appropriate alternative for sparse features

### Evaluation

- Primary metric: spectral W1 mean across 30 subgraphs per condition
- Primary statistics: paired t-test spectral vs gaussian (gate: spectral<gaussian with p<0.05)
- Secondary: spectral vs dropout, dropout vs gaussian

### Cross-dataset analysis

Regress `improvement_pct` on:
- `n_features_original`
- `feature_sparsity` (fraction of nonzeros)
- `energy_ratio` (max/min band energy of full graph at B=8)
- `mean_edge_density` across BFS subgraphs
- `svd_variance_explained_at_d16`

## Gate Criteria

- **Tier 1 (paper-critical):** Spectral augmentation wins on ≥3/4 datasets at p<0.05.
- **Tier 2 (publishable with caveats):** Spectral wins on ≥2/4 datasets; clear dataset-property predictor of when it helps.
- **Tier 3 (paper killed):** Spectral only wins on Cora. Effect is Cora-specific.

## Implementation plan

### Files to create

- `graph_fans/phase6/dataset_loader.py` — unified loader for Planetoid (Cora/CiteSeer/PubMed), Amazon (Photo/Computers), Coauthor (CS/Physics) via torch_geometric. Returns `GraphData` (existing dataclass).
- `scripts/test_phase7a_multidataset.py` — main experiment. Iterates over datasets × augmentations.
- `scripts/analyze_phase7a.py` — cross-dataset statistical analysis + plots.
- `tests/test_phase7.py` — smoke tests for new loaders.

### Files to modify

- `graph_fans/utils/graph_generators.py` — extend `load_citation_network` or add new functions for Amazon/Coauthor datasets (or just import from new `dataset_loader.py`).

### Compute budget

- 4 datasets × 30 subgraphs × 3 augmentations × 500 epochs ≈ 360 training runs
- At ~90s/run on L40S = 9h GPU
- Sparse eigendecomp (n=100) is fast (<1s)
- PCA fitting on large matrices (PubMed n=19717, Coauthor n=18333): cached once per dataset
- Total: ~10-12h GPU end-to-end

## Statistical power

Phase 6c-1 taught us n=30 subgraphs detects a ~14% effect at p=0.004. For Phase 7a we expect effect sizes from +5% to +60% based on Phase 6b. n=30 subgraphs per dataset is sufficient.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Spectral aug only helps Cora | Medium | Kills primary contribution | Tier 3 fallback: scope the paper to Cora, reframe as case study |
| Some datasets have subgraphs where GCN doesn't denoise | Low | Excludes that dataset | Run Phase 6a-style scale diagnostic first on each new dataset |
| Phase 6b +46% doesn't replicate at n=30 | Low | Weakens baseline | Larger effect should be easier to reproduce |
| Amazon/Coauthor loaders fail | Low | 1-2h debug | Use only Planetoid datasets (Cora + CiteSeer + PubMed) as fallback |

## Verification

1. Phase 6a-style sanity check per new dataset: GCN denoises at n=100 d=16, std_ratio < 3 for ≥75% of subgraphs.
2. Per-dataset summary table + forest plot of effect sizes
3. Cross-dataset regression plot + correlation report
4. Results written to `results/phase7a/`

## Expected paper outcome

- **Headline:** Spectral-structure-aware augmentation for single-graph diffusion: consistent improvement over gaussian across N real-world datasets
- **Result table:** 4 datasets × 3 augmentations with spectral showing X-Y% improvement consistently
- **Theory section:** spectral augmentation matches the diffusion model's Laplacian-eigenbasis assumption; explain mechanism
- **Ablation:** Phase 6a infrastructure details + Phase 2g as historical context (shaping shown to work on synthetic but narrow)

## Timeline

- Day 1: Implement dataset loaders + tests (~2h local)
- Day 2: Phase 6a-style sanity check on all 4 datasets (~2h GPU)
- Day 3: Full multi-dataset experiment (~10h GPU, probably split across 2-3 sessions)
- Day 4: Analysis + cross-dataset regression + report

## Gate decision mapping to next phase

- **Tier 1 met:** → Draft paper, target NeurIPS/ICML with Framing B primary
- **Tier 2 met:** → Add Phase 7b: characterize when spectral aug helps; may target workshop or KDD
- **Tier 3 met:** → Phase 7b explores downstream task utility as last chance to demonstrate practical value
