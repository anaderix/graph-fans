# Phase 3a: Per-Mode Noise Shaping — NO-GO

## Goal

Test whether per-mode noise shaping (n=50 individual eigenmode weights) improves upon per-band shaping (B=8 uniform-width bands, validated in Phase 2g) for spectral-aware graph feature generation. The hypothesis was that finer spectral resolution — matching the per-eigenmode granularity of the W1 evaluation metric — would amplify the 6-12% improvement seen in Phase 2g.

## Method

**3-way comparison:** uniform (no shaping) vs band-spectral (B=8, Phase 2g style) vs mode-spectral (n=50 per-eigenmode weights).

**Families tested:**
- SBM(q=0.05): 5 seeds (complete)
- SBM(q=0.1): 3 of 5 seeds (stopped early)
- BA(m=2) and SBM(q=0.01): not reached (experiment stopped at 24/60 runs)

**Training config:** 3-layer GCN, 128 hidden, 500 epochs, cosine SDE, EMA (0.999), LR annealing, epsilon-prediction + DDIM sampling. Identical to Phase 2g except for the addition of mode shaping.

**Evaluation:** Spectral Wasserstein-1 (per-eigenmode distributional distance), the standard metric since Phase 2f.

**Weights:** Both band and mode importance weights derived from seed-0 training split using the FANS formula: g_k = (pi_k + epsilon)^{-alpha}, normalized to mean=1.

- Band weights (SBM q=0.05): contrast range 1.4-10.9x (8 weights)
- Mode weights (SBM q=0.05): contrast range 0.016-1.9x before normalization, 120x dynamic range (50 weights)

**Statistical tests:** Paired t-test with Bonferroni correction (alpha=0.05/4=0.0125 per family).

## Results

### SBM(q=0.05) — 5 seeds

| Seed | Uniform W1 | Band W1 | Mode W1 | Band vs Uniform | Mode vs Uniform |
|------|-----------|---------|---------|-----------------|-----------------|
| 0 | 703.3 | 575.0 | 1056.5 | -18.2% | +50.2% |
| 1 | 629.3 | 548.7 | 1104.7 | -12.8% | +75.5% |
| 2 | 657.2 | 549.2 | 1266.5 | -16.4% | +92.7% |
| 3 | 807.3 | 710.0 | 1564.3 | -12.0% | +93.7% |
| 4 | 831.7 | 756.8 | 1312.9 | -9.0% | +57.8% |
| **Mean** | **725.8 +/- 80.5** | **627.9 +/- 87.9** | **1261.0 +/- 179.4** | **-13.5%** | **+73.7%** |

**Pairwise statistical tests (SBM q=0.05):**

| Comparison | Delta | t-stat | p-value | Significant? |
|-----------|-------|--------|---------|--------------|
| Uniform vs Band | -97.9 (13.5% better) | 10.16 | 0.00053 | **Yes** (alpha=0.0125) |
| Uniform vs Mode | +535.2 (73.7% worse) | -7.79 | 0.0015 | **Yes** (alpha=0.0125) |
| Band vs Mode | +633.1 (100.8% worse) | -9.39 | 0.00072 | **Yes** (alpha=0.0125) |

### SBM(q=0.1) — 3 seeds (incomplete)

| Seed | Uniform W1 | Band W1 | Mode W1 | Band vs Uniform | Mode vs Uniform |
|------|-----------|---------|---------|-----------------|-----------------|
| 0 | 1012.2 | 898.4 | 1572.7 | -11.2% | +55.4% |
| 1 | 1059.4 | 917.1 | 1623.1 | -13.4% | +53.2% |
| 2 | 1040.1 | 1009.4 | 1671.4 | -2.9% | +60.7% |
| **Mean** | **1037.2 +/- 23.9** | **941.6 +/- 60.1** | **1622.4 +/- 49.4** | **-9.2%** | **+56.4%** |

### Summary across families

| Method | SBM(q=0.05) Mean W1 | SBM(q=0.1) Mean W1 |
|--------|---------------------|---------------------|
| Uniform | 725.8 +/- 80.5 | 1037.2 +/- 23.9 |
| Band | 627.9 +/- 87.9 | 941.6 +/- 60.1 |
| Mode | 1261.0 +/- 179.4 | 1622.4 +/- 49.4 |

Band shaping improves W1 by 9-14% (replicating Phase 2g). Mode shaping worsens W1 by 56-74% — consistently and significantly worse than both uniform and band.

### Training loss vs generation quality

| Method | SBM(q=0.05) Mean Loss | SBM(q=0.05) Mean Std Ratio | SBM(q=0.05) Mean W1 |
|--------|----------------------|---------------------------|---------------------|
| Uniform | 0.421 | 1.68 | 725.8 |
| Band | 0.427 | 1.54 | 627.9 |
| Mode | 0.381 | 2.01 | 1261.0 |

Mode shaping achieves the **lowest training loss** (0.381 vs 0.421 uniform) but the **worst generation quality** (W1 1261 vs 726). This paradox is explained by the train/generate noise mismatch (see Root Cause below).

The std_ratio (gen_std / train_std) is also highest for mode shaping (2.01 vs 1.68 uniform), indicating the generated features deviate more from the training distribution.

### Per-band W1 breakdown (SBM q=0.05, seed 0 example)

| Band | Uniform W1 | Band W1 | Mode W1 |
|------|-----------|---------|---------|
| 0 (lowest freq) | 650.1 | 532.9 | 975.9 |
| 1 | 34.5 | 25.6 | 51.4 |
| 2 | 4.2 | 4.8 | 5.8 |
| 3 | 2.0 | 1.6 | 5.9 |
| 4 | 1.7 | 1.2 | 4.5 |
| 5 | 5.2 | 6.5 | 3.2 |
| 6 | 2.7 | 0.9 | 6.2 |
| 7 (highest freq) | 2.9 | 1.4 | 3.6 |

Mode shaping degrades W1 across nearly all bands, with the largest absolute impact in band 0 (the dominant low-frequency band containing community structure). The degradation is not confined to any particular spectral region.

## Root cause: train/generate noise distribution mismatch

The fundamental issue is a distributional mismatch between training and generation:

1. **During training:** Noise is shaped in the eigenbasis. For mode shaping, each eigenmode k receives noise scaled by sqrt(g_k), where g_k ranges from 0.016 to 1.9 (120x contrast). The score network learns to denoise this specifically shaped noise distribution.

2. **During generation (DDIM):** The initial noise x_T = torch.randn(...) is standard i.i.d. Gaussian — it has NOT been shaped. Every eigenmode receives equal noise variance. The score network, trained on shaped noise, receives unshaped noise and produces incorrect denoising predictions.

3. **Why band shaping survives but mode shaping does not:** Band shaping uses per-band empirical variance normalization after scaling. This partially masks the train/generate mismatch by keeping the overall variance structure closer to standard Gaussian within each band. Per-mode shaping applies raw sqrt(g_k) scaling without normalization (analytically correct for the Gaussian input, but creating a larger distributional gap). With 50 individual modes at 120x contrast range, the mismatch is far more severe than with 8 bands at 11x contrast.

4. **The training loss paradox:** Mode shaping gives lower training loss because the shaped noise is "easier" to predict — the model learns the non-uniform noise distribution. But at generation time, the noise is uniform, and the model's predictions are miscalibrated. Lower training loss does not translate to better generation quality when the noise distributions differ.

This is exactly the insight motivating Phase 3b (matched generation noise): shape the initial generation noise x_T to match the training noise distribution, eliminating the mismatch.

## Gate decision

**Phase 3a per-mode shaping: NO-GO**

Per-mode noise shaping is conclusively harmful:
- 50-94% W1 degradation across all 8 completed seed triples
- Statistically significant degradation (p=0.0015 for SBM q=0.05, all 5 seeds worse)
- Effect is consistent across 2 families (SBM q=0.05 and SBM q=0.1)
- The result was so clear that the experiment was stopped early at 24/60 runs (2 of 4 families)

## Implications

1. **Band shaping (Phase 2g) remains the best approach.** The 8-band discretization with per-band variance normalization provides both spectral awareness and robustness to the train/generate noise mismatch.

2. **Phase 3b (matched generation noise) is the natural next step.** By shaping x_T to match the training noise distribution, the mismatch that kills per-mode shaping could be eliminated. If successful, per-mode shaping could be revisited with matched generation noise.

3. **Granularity is not the bottleneck.** Moving from 8 bands to 50 modes did not improve quality — it made it dramatically worse. The issue is not spectral resolution but the consistency between training and generation noise distributions.

4. **Training loss is misleading when train/generate distributions differ.** This is a general cautionary finding for noise-shaping approaches in diffusion models: training metrics only predict generation quality when the training and generation noise distributions match.

## Experiment details

- **Script:** `scripts/test_phase3a_shaping_w1.py`
- **Results:** `results/phase3a/phase3a_results.json` (SBM q=0.05 complete), `results/phase3a/phase3a_run.log`
- **Datasets:** `results/phase3a/datasets/` (cached .npz files)
- **Runs completed:** 24/60 (SBM q=0.05: 15/15, SBM q=0.1: 9/15, BA m=2: 0/15, SBM q=0.01: 0/15)
- **Compute:** NVIDIA L40S GPU
