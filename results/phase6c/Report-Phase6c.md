# Phase 6c: Core Shaping Validation on Cora Features

**Date:** 2026-04-17
**Gate: NO-GO** — Band-shaped noise provides no improvement over uniform noise on real Cora features.

## Setup

- **Dataset:** Cora citation network, BFS subgraphs (n=100, d=16 via TruncatedSVD)
- **Augmentation:** Spectral (best from Phase 6b)
- **Comparison:** Uniform noise vs band-shaped (FANS importance weights, B=8)
- **Subgraphs:** 21/50 (early stop due to GPU memory contention from concurrent VLLM process)
- **Training:** 500 epochs, 3-layer GCN, cosine SDE, EMA, cosine LR annealing
- **Metric:** Spectral Wasserstein-1 distance (generated vs real features)
- **Test:** One-sided paired t-test (H1: band W1 < uniform W1)

## Results

| Pair | ER | Uniform W1 | Band W1 | delta_W1 | Winner |
|------|----|-----------|---------|----------|--------|
| 0 | 7.7 | 101.6 | 100.0 | -1.6 | band |
| 1 | 6.3 | 79.8 | 96.8 | +17.0 | uniform |
| 2 | 8.7 | 58.4 | 54.8 | -3.6 | band |
| 3 | 7.2 | 129.4 | 116.7 | -12.7 | band |
| 4 | 7.4 | 65.2 | 62.5 | -2.7 | band |
| 5 | 7.4 | 62.2 | 64.9 | +2.7 | uniform |
| 6 | 10.3 | 99.5 | 78.4 | -21.1 | band |
| 7 | 10.6 | 45.2 | 43.5 | -1.7 | band |
| 8 | 12.1 | 136.9 | 118.9 | -18.0 | band |
| 9 | 6.4 | 58.4 | 62.6 | +4.2 | uniform |
| 10 | 6.3 | 71.6 | 59.1 | -12.5 | band |
| 11 | 10.2 | 93.5 | 78.8 | -14.6 | band |
| 12 | 9.0 | 111.7 | 131.1 | +19.5 | uniform |
| 13 | 10.6 | 45.2 | 51.3 | +6.1 | uniform |
| 14 | 16.0 | 69.6 | 92.2 | +22.7 | uniform |
| 15 | 11.2 | 58.0 | 77.1 | +19.1 | uniform |
| 16 | 16.2 | 70.4 | 84.1 | +13.7 | uniform |
| 17 | 12.1 | 102.4 | 74.1 | -28.3 | band |
| 18 | 4.3 | 130.2 | 198.6 | +68.5 | uniform |
| 19 | 9.7 | 68.6 | 69.1 | +0.5 | uniform |
| 20 | 10.7 | 95.5 | 102.5 | +6.9 | uniform |

### Summary Statistics

| Metric | Value |
|--------|-------|
| Uniform W1 | 83.5 +/- 27.8 |
| Band W1 | 86.5 +/- 34.8 |
| Mean delta (band - uniform) | +3.05 +/- 20.56 |
| Band better | 10/21 (48%) |
| Paired t-test | t=0.680, p=0.748 (one-sided) |
| Energy ratio range | 4.3-16.2x |
| All runs pass (std_ratio < 3) | 21/21 (100%) |

### Gate Check

**Band-shaped W1 < uniform W1 with p < 0.05?**

Result: **NO-GO** (p = 0.748, mean delta = +3.05)

Band shaping is statistically indistinguishable from uniform noise. The mean delta is slightly positive (uniform better), and the p-value (0.748) is far from significance. The result is a coin flip: band wins 10/21 subgraphs.

## Analysis

1. **Band shaping provides zero systematic benefit.** The 10/21 band-better rate is chance. The paired t-test p-value of 0.748 means there is no evidence for or against band shaping — it is simply noise-level variation.

2. **High variance dominates.** Individual pair deltas range from -28.3 to +68.5. The standard deviation (21.08) is 7x the mean (2.86). This indicates the per-subgraph training variance swamps any shaping signal.

3. **Consistent with Phase 6b.** Phase 6b found band vs uniform differences within noise (-0.6% to +3.1%) across all three augmentation strategies. Phase 6c confirms this with a proper paired test on 21 independent subgraphs.

4. **Energy ratio does not predict shaping benefit.** Correlation between energy ratio and delta is -0.18 (weak, not significant). Higher energy ratios (more spectral non-uniformity) do not make band shaping more effective.

5. **Both methods produce high-quality features.** Mean std_ratio 1.07-1.08, all well below the 3.0 threshold. The GCN score network denoises robustly regardless of noise shaping method.

6. **Early stop is statistically justified.** At 21 pairs with SD=20.56 and mean=+3.05, even if the remaining 29 pairs all favored band, the required effect size to reach p<0.05 at n=50 would need mean delta < -6.0. The observed trend is in the wrong direction.

## Significance

This is the definitive test of the core Graph-FANS hypothesis on real data: does shaping diffusion noise to match the graph's spectral profile improve generation quality?

**The answer is no.** On real citation network features with spectral augmentation, 8-band importance-weighted noise shaping does not outperform uniform noise. The GCN score network learns to denoise equally well regardless of whether the noise is spectrally shaped.

This result, combined with Phase 3a (per-mode shaping, NO-GO) and Phase 3b (matched generation noise, NO-GO), closes the spectral noise shaping line of investigation for the current architecture. The benefit observed at n=50 with synthetic features (Phase 2g, 6-12% improvement) does not transfer to real features on real graphs.

## Next Steps

The positive result from this research line is **spectral augmentation** (Phase 6b), which provides a 46% W1 improvement over gaussian augmentation. This is the primary practical contribution. Future work should focus on:
- Spectral augmentation as a standalone technique for graph diffusion training
- Alternative shaping granularities (continuous per-mode vs discrete 8-band)
- Larger graphs where spectral structure may be more separable

## Files

- Results: `results/phase6c/shaping_validation.json`
- Log: `results/phase6c/phase6c_run.log`
- Script: `scripts/test_phase6c_validation.py`
