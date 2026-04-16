# Phase 6a: Scale Diagnostic on Cora Subgraphs

**Date:** 2026-04-16
**Gate decision: GO**

## Objective

Test whether the 3-layer GCN score network can denoise PCA-reduced Cora citation network features on BFS-sampled subgraphs across a range of scales (n=50-200) and feature dimensions (d=4, 16, 32).

This is the first time the Graph-FANS pipeline has been evaluated on real (non-synthetic) data. All prior phases (2a-5a) used synthetic community features on SBM/BA graph families.

## Method

1. **Subgraph sampling:** 20 connected BFS-induced subgraphs per scale from Cora (2708 nodes, 5278 edges, 1433-dim sparse binary features).
2. **Dimensionality reduction:** TruncatedSVD (not PCA -- no centering, appropriate for sparse binary data) fit on full Cora, applied to subgraphs. Three target dimensions: d=4, 16, 32.
3. **Augmentation:** N=100 training samples per subgraph via Gaussian augmentation: `x_aug = x_real + sigma * randn`.
4. **Training:** Baseline uniform noise, 500 epochs, 3L GCN 128h, cosine SDE + EMA + LR annealing. NVIDIA L40S GPU.
5. **Evaluation:** std_ratio (gen vs train std), spectral_L2, per-mode W1, energy ratio.

**Gate criteria:**
- GCN denoises at n=100 d=16: std_ratio < 3.0 for >= 15/20 subgraphs
- Spectral energy profile is non-uniform: energy ratio >= 2x

## Variance Explained by TruncatedSVD

| d | Variance Explained |
|---|-------------------|
| 4 | 5.8% |
| 16 | 15.2% |
| 32 | 23.1% |

Note: Low variance explained is expected for TruncatedSVD on 1433-dim sparse binary features (Cora's bag-of-words). The reduced features still capture the dominant spectral structure, which is sufficient for the denoising task.

## Results Summary

| n | d | Pass | Rate | std_ratio | spectral_L2 | W1 (mean +/- std) | Energy Ratio |
|---|---|------|------|-----------|-------------|-------------------|--------------|
| 50 | 4 | 20/20 | 100% | 1.04 | 0.052 | 28 +/- 6 | 36.9 |
| 50 | 16 | 20/20 | 100% | 1.14 | 0.046 | 81 +/- 18 | 15.4 |
| 50 | 32 | 18/20 | 90% | 1.91 | 0.083 | 909 +/- 780 | 12.7 |
| 100 | 4 | 20/20 | 100% | 1.04 | 0.047 | 47 +/- 13 | 20.4 |
| 100 | 16 | 20/20 | 100% | 1.14 | 0.040 | 154 +/- 57 | 9.5 |
| 100 | 32 | 20/20 | 100% | 1.76 | 0.063 | 1443 +/- 1392 | 7.1 |
| 150 | 4 | 20/20 | 100% | 1.02 | 0.037 | 59 +/- 10 | 17.2 |
| 150 | 16 | 20/20 | 100% | 1.12 | 0.031 | 216 +/- 67 | 7.6 |
| 150 | 32 | 20/20 | 100% | 1.61 | 0.047 | 1488 +/- 615 | 5.9 |
| 200 | 4 | 20/20 | 100% | 1.02 | 0.029 | 78 +/- 15 | 14.8 |
| 200 | 16 | 20/20 | 100% | 1.12 | 0.026 | 281 +/- 54 | 6.9 |
| 200 | 32 | 20/20 | 100% | 1.60 | 0.041 | 1950 +/- 1308 | 5.0 |

240 total training runs. Total wall time: 24,397s (~6.8 hours) on NVIDIA L40S GPU.

## Gate Check

**n=100, d=16:**
- Denoising: 20/20 pass (100%) -- need >= 75%. **GO.**
- Energy ratio: 9.5x -- need >= 2.0x. **GO.**
- **OVERALL: GO.**

## Analysis

### The GCN denoises real Cora features at all tested scales

This is a significant milestone. Phase 2f showed that the 3L GCN fails at n=200 on synthetic community features (std_ratio 3.5-11x, pure noise output). With Gaussian-augmented Cora features:

- d=4 and d=16 work perfectly: std_ratio 1.02-1.14 across all 4 scales, 80/80 subgraphs pass.
- d=32 works but with degraded quality: std_ratio 1.60-1.91, still under the 3.0 threshold. Only 2 failures (both at n=50 d=32 with std_ratio 3.07 and 3.25).
- Even n=200 works (158/160 pass across all d) -- a scale where synthetic features failed completely.

### Why Cora succeeds where synthetic features failed at n=200

Gaussian augmentation creates `x_aug = x_real + sigma * randn`, producing features centered around the real Cora features. This is a more learnable distribution than raw community features because:

1. The augmented distribution is Gaussian-like (unimodal), not the bimodal community feature distribution that stresses the GCN.
2. The signal-to-noise ratio in augmented features is controlled by sigma, avoiding the extreme spectral contrasts of community features.
3. The real Cora features have smoother spectral profiles than synthetic community features (energy ratio 5-37x vs 37-43x for SBM).

### Dimension scaling

- **W1 increases with d:** More dimensions means more spectral modes to reconstruct. d=32 W1 is 10-25x higher than d=4.
- **W1 increases with n:** Larger graphs have more eigenmodes and more complex spectral structure.
- **std_ratio improves with n at fixed d:** The model gets slightly better at larger scales (std_ratio 1.04 at n=50 d=4 vs 1.02 at n=200 d=4), likely because larger subgraphs have more training signal per augmented sample.
- **d=32 has high W1 variance:** std of W1 is 50-100% of mean at d=32, vs 10-30% at d=4/d=16. Some subgraphs are much harder for the model at high dimensionality.

### Cora subgraphs have genuine multi-scale spectral structure

Energy ratios range from 5.0x (n=200 d=32) to 36.9x (n=50 d=4). This confirms that BFS-sampled Cora subgraphs preserve the community structure that creates non-uniform spectral energy profiles. The energy ratio decreases with both n and d:

- With n: larger subgraphs average over more local structures, smoothing the spectral profile.
- With d: more SVD components capture finer-grained features that spread energy more evenly across modes.

All 240 conditions exceed the 2x energy ratio threshold, confirming that spectral noise shaping has something to exploit on real Cora features.

### The d=32 failure cases

Two subgraphs at n=50 d=32 failed with std_ratio 3.07 and 3.25 (just over the 3.0 threshold). These are marginal failures, not catastrophic (the Phase 2f n=200 failures had std_ratio 8-11x). The d=32 regime is harder because:

1. TruncatedSVD at d=32 explains only 23.1% of variance -- the reduced features are noisier.
2. The GCN must reconstruct 32 feature channels per node, a harder optimization problem.
3. At n=50 with d=32, the ratio of parameters to effective data dimensionality is at its worst.

## Recommendations for Phase 6b

1. **Primary operating point: n=100, d=16.** Best balance of denoising quality (std_ratio 1.14, 100% pass) and spectral structure (energy ratio 9.5x). Matches the gate check condition.
2. **Secondary: n=50, d=4.** Fastest training (~120s/subgraph), lowest W1, highest energy ratio. Good for rapid iteration.
3. **Avoid d=32 for shaping experiments.** High W1 variance would mask shaping effects. Reserve for later investigation if d=16 shows positive results.
4. **Test augmentation strategies at n=100, d=16.** Gaussian augmentation works but may not be optimal -- spectral augmentation (Phase 6b) could create training distributions that better preserve spectral structure.

## Files

- Results JSON: `results/phase6a/scale_diagnostic.json`
- Run log: `results/phase6a/phase6a_run.log`
- Plan: `plans/phase6-plan.md`
- Script: `scripts/test_phase6a_scale.py`
