# Phase 6b: Augmentation Strategy Comparison on Cora Subgraphs

**Date:** 2026-04-14
**Gate decision: GO**

## Objective

Compare three augmentation strategies (gaussian, spectral, dropout) crossed with two noise methods (uniform, band-shaped) on BFS-sampled Cora subgraphs to determine the best augmentation for Phase 6c shaping validation.

## Method

1. **Subgraph sampling:** 20 connected BFS-induced subgraphs from Cora at n=100 nodes.
2. **Dimensionality reduction:** TruncatedSVD d=16 (15.2% variance explained), fit on full Cora.
3. **Augmentation strategies:**
   - **Gaussian:** `x_aug = x_real + 0.1 * randn` (100 samples per subgraph)
   - **Spectral:** Project to eigenbasis, add energy-proportional noise (sigma=0.3)
   - **Dropout:** Random feature zeroing (drop_rate=0.2)
4. **Noise methods:** Uniform (baseline) and band-shaped (FANS importance weighting from augmented data).
5. **Training:** 500 epochs, 3L GCN 128h, cosine SDE + EMA + LR annealing. NVIDIA L40S GPU.
6. **Evaluation:** W1 distance (generated vs real un-augmented features), std_ratio sanity check.

**Gate criteria:** At least one augmentation with std_ratio < 3.0 for >= 75% of subgraphs.

## Results Summary

| Strategy | Noise | Pass | Rate | std_ratio | W1 (mean +/- std) | Loss |
|----------|-------|------|------|-----------|--------------------|------|
| spectral | uniform | 20/20 | 100% | 1.08 | 83.4 +/- 34.0 | 0.160 |
| spectral | band | 20/20 | 100% | 1.08 | 85.7 +/- 34.6 | 0.160 |
| dropout | uniform | 20/20 | 100% | 1.22 | 112.7 +/- 38.7 | 0.212 |
| dropout | band | 20/20 | 100% | 1.23 | 116.2 +/- 40.6 | 0.212 |
| gaussian | uniform | 20/20 | 100% | 1.14 | 154.6 +/- 57.2 | 0.202 |
| gaussian | band | 20/20 | 100% | 1.13 | 153.6 +/- 53.6 | 0.202 |

120 total training runs. Total wall time: 12,987s (~3.6 hours) on NVIDIA L40S GPU.
Energy ratios across subgraphs: 4.3-16.2x (mean 9.5x).

## Gate Check

**At least one augmentation with >= 75% pass rate?**
- All 6 conditions achieve 100% pass rate (120/120 pass). **GO.**
- Best strategy: **spectral/uniform** (W1=83.4).

## Analysis

### Spectral augmentation is the clear winner

Spectral augmentation achieves 46% lower W1 than gaussian and 26% lower than dropout. The ranking is consistent across all 20 subgraphs: spectral < dropout < gaussian.

Why spectral wins: it adds noise proportional to per-eigenmode energy, preserving the spectral structure of the real features. The augmented distribution stays close to the real feature manifold in spectral space, giving the score network a more faithful training signal. Gaussian augmentation adds isotropic noise that flattens spectral structure. Dropout preserves structure better than gaussian but introduces discontinuities.

### Band-shaped noise shows no advantage over uniform

For all three strategies, band vs uniform W1 differences are within noise:
- Gaussian: band -0.6% (negligible)
- Spectral: band +2.8% (slightly worse)
- Dropout: band +3.1% (slightly worse)

This is consistent with Phase 2g findings on synthetic data: band shaping at 8 bands is too coarse to capture the spectral structure that matters. The augmentation strategy dominates over the noise shaping method.

### All conditions produce stable, non-degenerate features

std_ratio ranges from 1.08 (spectral) to 1.23 (dropout), all well below the 3.0 threshold. No failures, no warnings. The GCN denoises successfully across all strategies and all 20 subgraph topologies.

### Spectral augmentation also achieves lowest training loss

Spectral augmentation loss (0.160) is lower than dropout (0.212) and gaussian (0.202). The augmented training distribution created by spectral augmentation is inherently easier for the epsilon-prediction objective, because the noise structure aligns with the Laplacian eigenbasis that the GCN operates in.

## Recommendations for Phase 6c

1. **Use spectral augmentation for Phase 6c.** Clear winner with 46% W1 improvement over the Phase 6a baseline (gaussian).
2. **Phase 6c should focus on whether per-mode shaping (not coarse band shaping) can improve W1.** The 8-band shaping tested here is too coarse, confirming the Phase 3a direction.
3. **The primary operating point remains n=100, d=16.** All 20 subgraphs pass at this scale.

## Files

- Results JSON: `results/phase6b/augmentation_comparison.json`
- Run log: `results/phase6b/phase6b_run.log`
- Script: `scripts/test_phase6b_augmentation.py`
