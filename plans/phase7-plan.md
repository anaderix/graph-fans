# Phase 7: Multi-Dataset Validation of Spectral Augmentation for Real-World Tasks

## Context

Phase 6 established the experimental infrastructure and closed the band-shaping story on real data: band shaping works on synthetic block-structured graphs (SBM/BA) but fails on Cora (Phase 6c-1 Part B: -14.4%, p=0.004). The practical contribution from Phase 6 is **spectral augmentation** (Phase 6b: +46% W1 improvement on Cora over gaussian augmentation).

Phase 7 validates spectral augmentation on datasets with real downstream tasks — specifically label-scarce node classification where augmentation has immediate practical value. We abandon Cora/PubMed/Coauthor as the primary datasets (weak "so what?" motivation: nobody needs synthetic academic papers) in favor of domains with genuine stakeholder interest: **financial fraud detection** and **protein function prediction**.

## Paper framing target

**Primary claim:** Spectral feature augmentation — generating training variants by adding noise in the Laplacian eigenbasis proportional to per-mode energy — preserves the spectral structure GNNs rely on and outperforms generic augmentation (DropFeature, FeatMask, GraphMix) on label-scarce node classification benchmarks.

**Secondary claim:** The approach is naturally privacy-preserving and fits federated learning settings where raw feature sharing is restricted.

## Sub-Phases

---

### Phase 7-pre: Continuous-Feature Sanity Check (0.5-1 day)

**Objective:** Confirm the Phase 6 pipeline (BFS subgraphs + TruncatedSVD + augmentation + 3L GCN) works on datasets with continuous features. All prior positive results were on Cora's sparse binary bag-of-words features (15.2% variance explained at d=16). Elliptic Bitcoin (continuous transaction statistics) and Stanford PPI (continuous protein features) are the actual target domains — if the pipeline collapses on continuous data, the whole multi-dataset validation fails.

**Datasets (subgraph sanity only):**
- **Elliptic Bitcoin** — 203k nodes, 166 continuous features (aggregated transaction statistics), illicit/licit/unknown labels
- **Stanford PPI** — ~57k proteins (across 24 tissue PPIs), 50 continuous features (positional/motif), 121 multi-label functions

**Method:**
1. Add loaders for Elliptic (via PyTorch Geometric `EllipticBitcoinDataset`) and PPI (via PyG `PPI` or direct from Stanford BioSNAP).
2. BFS-sample 10 subgraphs at n=100 from each dataset's largest connected component.
3. TruncatedSVD fit on full feature matrix → d=16 (match Phase 6b operating point).
4. Report per-subgraph:
   - SVD variance explained at d=16
   - Spectral energy ratio (max/min band energy at B=8)
   - For 3 representative subgraphs: train baseline (uniform noise, Gaussian augmentation, 500 epochs) and measure std_ratio, spectral_L2, W1 vs un-augmented features
5. Check that the 3L GCN denoises at n=100 d=16 (std_ratio < 3.0 per Phase 6a gate).

**Files to create:**
- `graph_fans/phase7/__init__.py`
- `graph_fans/phase7/real_dataset_loader.py` — unified PyG loader for Elliptic + PPI + fallback datasets
- `scripts/test_phase7pre_sanity.py`
- Extend `tests/test_phase6.py` or add `tests/test_phase7.py` with loader smoke tests

**Reuses:** existing Phase 6 infrastructure (subgraph_sampler, feature_reducer, augmentation, Trainer, spectral_w1_summary).

**Gate criteria:**
- Both datasets load and produce valid `GraphData` objects.
- SVD variance explained at d=16 ≥ 20% on both (Cora was 15.2%; continuous features should do better).
- GCN denoises on ≥ 2/3 sanity subgraphs per dataset (std_ratio < 3.0).
- Gaussian augmentation produces generated features with W1 < 5× the training variance.

**If gate fails:**
- If Elliptic fails but PPI works → drop Elliptic, use PPI + a backup (T-Finance or DGraph-Fin).
- If both fail → pipeline doesn't handle continuous features; need to investigate. Options: test dense non-reduced features (skip SVD), different d, different architecture.
- If SVD variance is very low → the d=16 reduction may be throwing away most of the signal; re-evaluate dimensionality.

**Effort:** ~2 hours implementation + 1 hour GPU + 1 hour analysis.

**Depends on:** Phase 6a/6b infrastructure (already present).

---

### Phase 7a: Cross-Dataset Spectral Augmentation Validation (1-2 days)

**Objective:** Replicate Phase 6b's finding (spectral > gaussian augmentation in W1) across multiple domains with real downstream applications. Establish generalization beyond Cora.

**Datasets:**
- **Primary: Elliptic Bitcoin** (fraud detection)
- **Secondary: Stanford PPI** (protein function prediction)
- **Tertiary (if time): DGraph-Fin** (industrial fraud benchmark)
- **Baseline reference: Cora** (keep for continuity with Phase 6b)

**Method:**
For each dataset:
1. BFS-sample 30 subgraphs at n=100 (matches Phase 6c-1 Part B power level).
2. TruncatedSVD fit on full features → d=16.
3. For each subgraph × augmentation strategy (gaussian / spectral / dropout) × noise (uniform only; no band shaping — deprecated for real data):
   a. Augment single subgraph feature matrix → 100 training samples.
   b. Create separate reference set of 50 augmented samples (different seed).
   c. Train 3L GCN, 500 epochs, cosine+EMA.
   d. Generate 50 samples, compute spectral W1 vs un-augmented real features.
4. Paired t-test across 30 subgraphs per dataset per augmentation pair.

**Conditions per dataset: 3 augmentations × 30 subgraphs = 90 training runs.**

**Evaluation:**
- Primary: spectral W1 mean across 30 subgraphs per augmentation
- Statistics: paired t-test spectral vs gaussian (two-sided; gate: spectral < gaussian at p < 0.05)
- Secondary: spectral vs dropout, dropout vs gaussian
- Effect size: percent improvement relative to gaussian baseline

**Cross-dataset analysis (novel contribution):**
Regress `improvement_pct(spectral vs gaussian)` on dataset properties:
- `n_features_original` (sparsity)
- `feature_sparsity_fraction` (nonzero ratio)
- `svd_variance_explained_at_d16`
- `energy_ratio` (spectral non-uniformity)
- `mean_edge_density` across subgraphs
- `intra_class_feature_variance` (a property unique to labeled datasets)

Produces a figure: effect size vs dataset property, with interpretation of when spectral augmentation helps most.

**Gate criteria (three tiers):**
- **Tier 1 (paper-ready):** Spectral wins (p < 0.05) on ≥ 3/4 datasets.
- **Tier 2 (paper with caveats):** Spectral wins on ≥ 2/4 datasets; a dataset-property predictor of when it helps.
- **Tier 3 (paper killed):** Spectral wins only on Cora.

**Files to create:**
- `scripts/test_phase7a_multidataset.py` (main experiment)
- `scripts/analyze_phase7a.py` (cross-dataset regression + plots)
- `graph_fans/phase7/augmentation_benchmarks.py` — wrapper for DropFeature/FeatMask/etc. for Phase 7b

**Reuses:** Phase 6b augmentation strategies (already implemented in `graph_fans/phase6/augmentation.py`).

**Effort:** ~9-12 hours GPU + analysis.

**Depends on:** Phase 7-pre gate pass on at least Elliptic or PPI.

---

### Phase 7b: Downstream Task Payoff (2-3 days)

**Objective:** Show that spectral augmentation improves **real classification accuracy** on the chosen benchmarks, not just spectral W1. This is the "does this matter" experiment that separates a methodological contribution from a practically useful one.

**The payoff test:** Phase 2g showed 12.5% W1 improvement producing only 0.9% downstream accuracy gain. We need to beat that pattern — if spectral augmentation gives <1% accuracy improvement in label-scarce settings, the paper reduces to a metric-only result. A +3-5% improvement over best prior augmentation would be a strong contribution.

**Setup per dataset:**
1. Use the full graph (not subgraphs) with natural label splits where available.
2. Vary label fraction: {1%, 5%, 10%, 20%, full}.
3. For each label fraction, train a standard classifier (GCN / GAT / SGC) with:
   - No augmentation (vanilla baseline)
   - DropFeature (Rong 2020)
   - FeatMask
   - GraphMix / mixup (You 2020)
   - FLAG (adversarial, Kong 2022)
   - Our: **spectral augmentation of training features**
4. Evaluate on held-out test set.
   - Elliptic: F1 on illicit class (minority); AUC-PR
   - PPI: multi-label micro-F1

**Seeds:** 10 random seeds per (dataset × label_fraction × augmentation) → measure mean ± CI.

**Statistical test:** Paired comparison (same seed) of spectral augmentation vs best of {gaussian, DropFeature, FeatMask, GraphMix, FLAG}.

**Files to create:**
- `graph_fans/phase7/downstream_classifier.py` — standard GCN/GAT/SGC classifier
- `graph_fans/phase7/augmentation_baselines.py` — DropFeature, FeatMask, GraphMix, FLAG
- `scripts/test_phase7b_downstream.py`
- `scripts/analyze_phase7b.py` — plots of accuracy vs label fraction

**Gate criteria:**
- **Strong win (publishable at NeurIPS/ICML):** Spectral augmentation beats best prior augmentation by ≥ 2% absolute F1/accuracy on ≥ 1 dataset at the 1-5% label fraction regime.
- **Weak win (workshop or KDD):** Spectral augmentation matches best prior on ≥ 2 datasets with additional benefit (e.g., lower variance across seeds).
- **No win (paper is methodological):** No consistent accuracy improvement; paper reduces to "we introduce spectral augmentation and show it improves spectral W1."

**Effort:** ~15-20 hours GPU + analysis (many configurations).

**Depends on:** Phase 7a gate pass (Tier 1 or Tier 2).

---

### Phase 7c (optional): Federated Fraud Detection (2 days)

**Objective:** Demonstrate the privacy-preserving application story. Fraud detection is inherently federated — multiple financial institutions have their own transaction graphs and cannot share raw data.

**Setup:**
1. Partition Elliptic Bitcoin into K clients (e.g., K=10, by time-slice or random subgraph).
2. Each client trains a local classifier on its partition with:
   - Raw features (baseline)
   - Spectral-augmented training features (our method)
3. Federated averaging of classifier parameters (standard FedAvg).
4. Evaluate global model on held-out transactions.

**Comparison:** federated spectral augmentation vs federated without augmentation vs centralized with augmentation (upper bound).

**Privacy angle:** quantify information leakage via DP-SGD compatibility — spectral augmentation adds noise to training data before gradient computation, which composes with DP-SGD noise for privacy accounting.

**Gate:** Spectral augmentation improves federated fraud detection F1 by ≥ 1% over federated baseline. Bonus: privacy budget cost is not materially higher.

**Effort:** ~8-10 hours GPU + privacy analysis.

**Depends on:** Phase 7b strong win on Elliptic.

---

## Compute budget summary

| Phase | Runs | GPU hours | Clock |
|-------|------|-----------|-------|
| 7-pre | ~30 | 1-2 | 0.5 day |
| 7a | 120-360 | 9-12 | 1-2 days |
| 7b | 150-200 | 15-20 | 2-3 days |
| 7c (optional) | 40-80 | 8-10 | 1-2 days |
| **Total** | **~400-700** | **~35-45** | **~1 week** |

All on NVIDIA L40S. Phase 6a's 240 runs on the same hardware took ~7 hours; Phase 7 scale is ~3-4× that.

## Paper outcome mapping

- **7a Tier 1 + 7b strong win:** Submit to NeurIPS / ICML / KDD with Framing B primary, full story (method + cross-dataset + downstream utility).
- **7a Tier 2 + 7b weak win:** Workshop submission (e.g., GLFrontiers, GraphCon), honest characterization paper.
- **7a Tier 3 or 7b no win:** Project stalls. Either pivot to theoretical analysis (why spectral augmentation helps W1 but not accuracy) or accept a negative result.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Continuous features break the pipeline (7-pre fails) | Low-Medium | Blocks everything downstream | Diagnose: try larger d, different architecture, full-graph training |
| Elliptic PyG loader has API changes | Low | Small delay | Use the raw dataset; many github repos have loaders |
| Spectral aug wins on W1 but not accuracy | Medium | Kills Framing B's strong story | Phase 7b is the gate — if W1 improvement is real but doesn't translate, paper is methodological only |
| Effect is Cora-specific (Tier 3) | Medium-High | Paper is local case study | Reframe as "spectral augmentation helps in graphs with property X" |
| Label-scarce regime doesn't favor augmentation over semi-supervised methods | Medium | Baselines beat us | Compare against full semi-supervised pipeline, not just augmentation methods |
| Federated story is weak because aggregation destroys augmentation benefit | Medium | 7c is lost | Skip 7c, focus on 7a/7b |

## Verification per sub-phase

- **7-pre:** Sanity plots (std_ratio histogram, spectral profile of PCA-reduced features) per dataset. Gate decision logged.
- **7a:** Summary table matching Phase 6b format (dataset × augmentation, W1 mean ± std, improvement %, p-value) + cross-dataset regression plot.
- **7b:** Forest plot of effect sizes across datasets and label fractions + confidence intervals + paired test p-values.
- **7c:** Federated vs centralized accuracy curves + privacy-utility tradeoff.

## Files to write in order

1. `graph_fans/phase7/real_dataset_loader.py` (7-pre)
2. `scripts/test_phase7pre_sanity.py` (7-pre)
3. `scripts/test_phase7a_multidataset.py` (7a)
4. `graph_fans/phase7/augmentation_baselines.py` (7b)
5. `graph_fans/phase7/downstream_classifier.py` (7b)
6. `scripts/test_phase7b_downstream.py` (7b)
7. `graph_fans/phase7/federated.py` (7c, optional)
8. `scripts/test_phase7c_federated.py` (7c, optional)

Tests added to `tests/test_phase7.py` at each sub-phase.

## Key references to lean on

- Augmentation baselines: DropFeature/FeatMask (Rong 2020), GraphMix/G-Mixup, FLAG (Kong 2022)
- Datasets: Elliptic (Weber 2019), Stanford PPI (Hamilton 2017, GraphSAGE), DGraph-Fin (Huang NeurIPS 2022)
- Fraud GNN baselines: Evolve-GCN (Pareja 2020), TGN, CARE-GNN (Dou 2020)
- Classifier backbones: GCN (Kipf 2017), GAT (Velickovic 2018), SGC (Wu 2019)

See `plans/phase7-related-work.md` (to be filled from deep research) for full citation list.
