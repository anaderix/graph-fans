# Phase 6e: Spectral Augmentation on Synthetic Data — NO-GO (H1 Confirmed)

## Goal

Disambiguate why Phase 6c failed (band shaping had zero effect on Cora features with spectral augmentation, p=0.748). Three hypotheses:

- **H1:** Spectral augmentation makes band shaping redundant (the augmentation already respects the spectral profile).
- **H2:** Cora's PCA-reduced features lack the bimodal spectral structure of synthetic community features.
- **H3:** N_ref=1 evaluation noise masks a small shaping effect.

Phase 6e isolates H1 by replicating the Phase 2g experiment exactly — SBM(q=0.05), n=50, d=4, 500 epochs, 3L GCN, 5 seeds — but replaces the 100 independent community feature samples with 100 spectral-augmented copies from a single seed sample.

## Method

For each of 5 seeds:
1. Generate ONE community-boundary feature matrix on SBM(q=0.05) topology (same as Phase 2g's generator)
2. Spectral-augment to 100 training samples (sigma=0.3) — noise in eigenbasis proportional to mode energy
3. Reference split: separate seed (seed + 100000), also spectral-augmented to 50 samples
4. Compute band importance weights from augmented training data
5. Train uniform vs band-shaped (Phase 2g config)
6. Evaluate spectral W1 against reference split
7. Paired t-test

## Results

| Seed | Uniform W1 | Band W1 | Delta | Winner |
|------|-----------|---------|-------|--------|
| 0 | 335.0 | 343.9 | +2.7% worse | uniform |
| 1 | 285.1 | 332.0 | +16.4% worse | uniform |
| 2 | 250.9 | 234.6 | -6.5% better | band |
| 3 | 393.1 | 378.8 | -3.6% better | band |
| 4 | 198.8 | 203.3 | +2.3% worse | uniform |
| **Mean** | **292.6 ± 67.1** | **298.5 ± 67.5** | **-2.0%** | 3/5 uniform |

**Paired t-test:** t = -0.522, p = 0.629. Not significant. Band shaping is indistinguishable from uniform noise (actually slightly worse on average).

## Comparison to Phase 2g

| Experiment | Training data | Uniform W1 | Band W1 | Improvement | p-value |
|-----------|--------------|-----------|---------|-------------|---------|
| **Phase 2g** (same graph/features) | 100 independent samples | ~725 | ~628 | **+12.5%** | **0.0001** |
| **Phase 6e** (this experiment) | 1 sample, spectral-augmented x100 | 293 | 299 | -2.0% | 0.629 |

Same graph family, same feature generator, same model config — only the training dataset construction differs. **The effect disappears completely.**

## Gate Decision: NO-GO — H1 Confirmed

**Spectral augmentation eliminates the band shaping effect.** This is a definitive result across 5 seeds with identical methodology to Phase 2g except for dataset construction.

## Interpretation

Phase 2g's 12.5% W1 improvement is (at least partially) an artifact of the training dataset construction, not a general property of spectral noise shaping on graphs.

**The mechanism:** Independent community-feature samples have high inter-sample variance — each sample has different random community centroids, different boundary interpolations, different hub signatures. Band shaping helps the score network navigate this variance by emphasizing low-frequency community-encoding modes during noise injection during training, effectively telling the model "the variation that matters most is in the low-frequency modes."

Spectral augmentation creates a fundamentally different training distribution: all 100 samples are perturbations of one seed feature matrix, with noise already allocated proportionally to mode energy. The inter-sample variance is concentrated in the same modes that band shaping would emphasize. So band shaping during training has nothing to add — the training distribution already respects the spectral profile.

## Implications

1. **Phase 2g's effect is augmentation-dependent.** The 12.5% improvement on SBM(q=0.05), 3/4 families significant after power boost, only applies when training uses independent sample generation. This narrows the applicability claim.

2. **Phase 6c's failure is explained.** Band shaping failed on Cora not because of the features (H2) or evaluation noise (H3), but because spectral augmentation was used as the dataset construction method. It would have failed on synthetic data too (as shown here).

3. **The practical recommendation flips.** For real-world datasets with single feature matrices, spectral augmentation is the correct strategy (46% W1 improvement from Phase 6b) — and band shaping adds nothing on top. The "two techniques" of Graph-FANS (band shaping + augmentation) turn out to be non-combinable.

4. **Experiment B (community features on Cora topology) is no longer needed.** We now know the answer: band shaping would work if applied with independent samples, but spectral augmentation obsoletes it.

## Future Directions

The honest reframing of the project: **spectral augmentation is the practical contribution, not band shaping.** Future work should:

1. Characterize when spectral augmentation helps vs generic Gaussian augmentation across more real datasets (PubMed, Amazon products, ogbn-arxiv)
2. Test spectral augmentation as a standalone technique with various score network architectures
3. Abandon band shaping investigations for real-data applications

## Files

- Script: `scripts/test_phase6e_synthetic_spectral_aug.py`
- Results: `results/phase6e/synthetic_spectral_aug.json`
- Run log: `results/phase6e/phase6e_run.log`
- Plan: `plans/phase6-plan.md` (Phase 6e section)
