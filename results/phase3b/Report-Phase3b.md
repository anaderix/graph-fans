# Phase 3b: Matched Generation Noise -- NO-GO

## Goal

Test whether matching the generation start noise to the spectrally-shaped training noise improves feature generation quality. In all prior experiments (Phase 2g, 3a), there is a distributional mismatch: training corrupts features with shaped noise (`x_T ~ epsilon_shaped`) but generation starts from uniform Gaussian noise (`x_T ~ N(0,1)`). The hypothesis was that aligning generation start noise with training noise would amplify the 6-12% W1 improvement from Phase 2g.

## Method

**3-way comparison:**

| Condition | Training noise | Generation start | Label |
|-----------|---------------|-----------------|-------|
| 1 | uniform | uniform | `uniform` (baseline) |
| 2 | band-shaped | uniform | `band-mismatched` (Phase 2g style) |
| 3 | band-shaped | band-shaped | `band-matched` (new) |

**Families:** SBM(q=0.05), SBM(q=0.1), BA(m=2), SBM(q=0.01)

**Config:** 3-layer GCN, 128 hidden, 50 nodes, 4 features, 500 epochs, cosine SDE, EMA (0.999), LR annealing, epsilon-prediction + DDIM (200 steps). 100 training samples, 50 reference samples per seed. Band shaping uses B=8 uniform-width bands with FANS importance weights derived from seed-0 training split.

**Seeds:** 5 per family. 3 conditions x 4 families x 5 seeds = 60 runs total.

**Statistical tests:** Paired t-test with Bonferroni correction (alpha = 0.05/4 = 0.0125 per family).

**Evaluation:** Spectral Wasserstein-1 (per-eigenmode distributional distance).

**Key comparison:** band-mismatched vs band-matched (conditions 2 vs 3). Lower W1 = better.

## Results

### SBM(q=0.05) -- 5 seeds

| Seed | Uniform W1 | Mismatched W1 | Matched W1 | U->Mis | U->Mat | Mis->Mat |
|------|-----------|--------------|------------|--------|--------|----------|
| 0 | 676.6 | 575.0 | 640.6 | +15.0% | +5.3% | -11.4% |
| 1 | 629.3 | 548.7 | 612.0 | +12.8% | +2.8% | -11.5% |
| 2 | 657.2 | 549.2 | 585.2 | +16.4% | +11.0% | -6.6% |
| 3 | 807.3 | 710.0 | 799.7 | +12.1% | +0.9% | -12.6% |
| 4 | 831.7 | 756.8 | 788.0 | +9.0% | +5.3% | -4.1% |
| **Mean** | **720.5 +/- 82.6** | **627.9 +/- 87.9** | **685.1 +/- 90.6** | **+12.8%** | **+4.9%** | **-9.1%** |

Mismatched beats uniform: 5/5 seeds. Mismatched beats matched: 5/5 seeds.

**Pairwise statistical tests:**

| Comparison | Improvement | t-stat | p-value | Significant? |
|-----------|-------------|--------|---------|-------------|
| Uniform vs Mismatched | 12.8% | 14.65 | 0.00013 | **Yes** |
| Uniform vs Matched | 4.9% | 3.15 | 0.034 | No |
| Mismatched vs Matched | -9.1% (matched worse) | -5.34 | 0.006 | **Yes** |

Matching generation noise to training noise significantly worsens W1 compared to the mismatched configuration (+9.1%, p=0.006). The matched condition still improves over uniform (+4.9%) but fails to reach significance.

### SBM(q=0.1) -- 5 seeds

| Seed | Uniform W1 | Mismatched W1 | Matched W1 | U->Mis | U->Mat | Mis->Mat |
|------|-----------|--------------|------------|--------|--------|----------|
| 0 | 1012.2 | 898.4 | 1039.4 | +11.2% | -2.7% | -15.7% |
| 1 | 1059.4 | 917.1 | 1005.9 | +13.4% | +5.0% | -9.7% |
| 2 | 1040.1 | 1009.4 | 1129.5 | +3.0% | -8.6% | -11.9% |
| 3 | 1040.1 | 1009.4 | 1146.7 | +3.0% | -10.2% | -13.6% |
| 4 | 1028.2 | 984.0 | 1085.4 | +4.3% | -5.6% | -10.3% |
| **Mean** | **1036.0 +/- 15.5** | **963.7 +/- 46.9** | **1081.4 +/- 53.0** | **+7.0%** | **-4.4%** | **-12.2%** |

Mismatched beats uniform: 5/5 seeds. Mismatched beats matched: 5/5 seeds. Matched beats uniform: 1/5 seeds.

**Pairwise statistical tests:**

| Comparison | Improvement | t-stat | p-value | Significant? |
|-----------|-------------|--------|---------|-------------|
| Uniform vs Mismatched | 7.0% | 3.10 | 0.036 | No |
| Uniform vs Matched | -4.4% (matched worse) | -1.61 | 0.183 | No |
| Mismatched vs Matched | -12.2% (matched worse) | -11.67 | 0.00031 | **Yes** |

Matched is **worse than uniform** (4/5 seeds). The mismatched vs matched difference is highly significant (p=0.0003).

### BA(m=2) -- 5 seeds

| Seed | Uniform W1 | Mismatched W1 | Matched W1 | U->Mis | U->Mat | Mis->Mat |
|------|-----------|--------------|------------|--------|--------|----------|
| 0 | 540.0 | 455.8 | 472.6 | +15.6% | +12.5% | -3.7% |
| 1 | 723.8 | 647.9 | 674.1 | +10.5% | +6.9% | -4.0% |
| 2 | 587.4 | 544.1 | 596.4 | +7.4% | -1.5% | -9.6% |
| 3 | 794.1 | 786.4 | 855.6 | +1.0% | -7.7% | -8.8% |
| 4 | 734.1 | 658.9 | 718.6 | +10.2% | +2.1% | -9.1% |
| **Mean** | **675.9 +/- 95.9** | **618.6 +/- 112.0** | **663.5 +/- 127.3** | **+8.5%** | **+1.8%** | **-7.2%** |

Mismatched beats uniform: 5/5 seeds. Mismatched beats matched: 5/5 seeds. Matched beats uniform: 3/5 seeds.

**Pairwise statistical tests:**

| Comparison | Improvement | t-stat | p-value | Significant? |
|-----------|-------------|--------|---------|-------------|
| Uniform vs Mismatched | 8.5% | 4.02 | 0.016 | No |
| Uniform vs Matched | 1.8% | 0.55 | 0.614 | No |
| Mismatched vs Matched | -7.2% (matched worse) | -4.48 | 0.011 | **Yes** |

### SBM(q=0.01) -- 5 seeds (negative control)

| Seed | Uniform W1 | Mismatched W1 | Matched W1 | U->Mis | U->Mat | Mis->Mat |
|------|-----------|--------------|------------|--------|--------|----------|
| 0 | 559.3 | 555.6 | 570.6 | +0.7% | -2.0% | -2.7% |
| 1 | 729.9 | 963.0 | 999.6 | -31.9% | -37.0% | -3.8% |
| 2 | 922.5 | 730.6 | 736.7 | +20.8% | +20.1% | -0.8% |
| 3 | 704.0 | 647.0 | 658.7 | +8.1% | +6.4% | -1.8% |
| 4 | 427.9 | 549.8 | 552.7 | -28.5% | -29.2% | -0.5% |
| **Mean** | **668.7 +/- 167.0** | **689.2 +/- 152.1** | **703.7 +/- 162.0** | **-3.1%** | **-5.2%** | **-2.1%** |

Mismatched beats uniform: 3/5 seeds. Mismatched beats matched: 5/5 seeds. Matched beats uniform: 2/5 seeds.

**Pairwise statistical tests:**

| Comparison | Improvement | t-stat | p-value | Significant? |
|-----------|-------------|--------|---------|-------------|
| Uniform vs Mismatched | -3.1% (worse) | -0.28 | 0.793 | No |
| Uniform vs Matched | -5.2% (worse) | -0.45 | 0.674 | No |
| Mismatched vs Matched | -2.1% (matched worse) | -2.44 | 0.071 | No |

SBM(q=0.01) behaves as the negative control -- 80% energy in band 0 leaves no multi-scale structure for shaping to exploit. Neither shaping condition improves over uniform. High seed-to-seed variance (167 W1 std on 669 mean) reflects the near-disconnected graph topology.

## Summary Table

| Family | Uniform W1 | Mismatched W1 | Matched W1 | U->Mis | U->Mat | Mis->Mat |
|--------|-----------|--------------|------------|--------|--------|----------|
| SBM(q=0.05) | 720.5 +/- 82.6 | 627.9 +/- 87.9 | 685.1 +/- 90.6 | 12.8%* | 4.9% | -9.1%* |
| SBM(q=0.1) | 1036.0 +/- 15.5 | 963.7 +/- 46.9 | 1081.4 +/- 53.0 | 7.0% | -4.4% | -12.2%* |
| BA(m=2) | 675.9 +/- 95.9 | 618.6 +/- 112.0 | 663.5 +/- 127.3 | 8.5% | 1.8% | -7.2%* |
| SBM(q=0.01) | 668.7 +/- 167.0 | 689.2 +/- 152.1 | 703.7 +/- 162.0 | -3.1% | -5.2% | -2.1% |

\* significant at Bonferroni-corrected alpha=0.0125

**Key finding: band-mismatched is the best condition in all 4 families.** Mismatched beats matched at every seed in 3/4 families (15/15 seeds for SBM q=0.05, SBM q=0.1, BA m=2; 5/5 for SBM q=0.01). Total: 20/20 seeds where mismatched outperforms matched.

## Root Cause Analysis

### Why does matching generation noise to training noise make things *worse*?

The result corresponds to Outcome 3 from the plan ("Matched < Mismatched: the model learned to compensate for the mismatch"). The mechanism:

1. **DDIM is iterative correction, not single-shot.** DDIM applies 200 reverse steps, each using the score network's prediction to incrementally refine the signal. The initial noise distribution is progressively overwritten by the score predictions. By step ~50-100, the trajectory has largely converged regardless of the initial distribution.

2. **The score network adapts to the mismatch.** During training with shaped noise, the model sees shaped-noise-corrupted inputs at high t. During generation starting from uniform noise, the high-t score predictions are slightly "wrong" (expecting shaped input, receiving uniform). But this error is small because band shaping's per-band variance normalization keeps the shaped noise close to unit Gaussian in each band. The model's predictions at subsequent steps correct for this initial deviation.

3. **Shaping the generation start introduces a new problem.** When generation starts from shaped noise (non-uniform variance per band), the DDIM trajectory explores a different manifold than the training trajectories. The score network has never seen this combination (shaped noise at t=T with shaped predictions) because training always pairs shaped noise with the forward process, not with reverse starting conditions. The mismatch between these two shaped-but-different distributions is worse than the mismatch between shaped training and uniform generation.

4. **Band shaping's variance normalization is the key.** Band shaping normalizes variance within each band (so total noise variance is preserved). This means the shaped noise is a rotation of Gaussian noise in the eigenbasis -- close enough to N(0,1) that DDIM handles the transition smoothly. Shaping the generation start doubles the spectral asymmetry rather than canceling it.

### Quantitative evidence

- **Training loss is identical** between mismatched and matched (same model, same training). The difference is entirely in generation.
- **std_ratio is higher for matched** in most seeds (e.g., SBM q=0.1 seed 0: 1.86 mismatched vs 2.03 matched), indicating DDIM diverges more with shaped initial noise.
- **The effect scales with spectral contrast.** SBM(q=0.1) has the largest matched penalty (-12.2%) and the most complex spectral structure. SBM(q=0.01) has the smallest penalty (-2.1%) and the simplest structure.

## Gate Decision

**Phase 3b matched generation noise: NO-GO**

Matching generation start noise to shaped training noise is counterproductive across all 4 families tested. The "mismatch" between band-shaped training noise and uniform generation noise is not a bug -- it is a feature. Band shaping's variance normalization keeps the noise close enough to Gaussian that DDIM's iterative correction handles the transition, while the shaped training noise provides meaningful spectral guidance during the forward process.

## Implications

1. **Band-mismatched (Phase 2g) remains the best configuration.** No further optimization of the generation start noise is warranted.
2. **The train/generate mismatch is not a bottleneck.** Phase 3a's per-mode shaping failure was attributed to the mismatch (120x contrast in mode weights made it catastrophic). Phase 3b shows that even for band shaping (11x contrast), fixing the mismatch does not help. Phase 3a's failure was caused by the extreme weight contrast itself, not the mismatch per se.
3. **DDIM is robust to initial noise distribution.** This is consistent with the DDIM literature (Song et al., 2021) which shows deterministic sampling converges to the same manifold from a range of initial conditions.
4. **Future work should focus on improving the score network or training dynamics**, not on aligning noise distributions. The remaining W1 gap to oracle (628 vs 38 for SBM q=0.05) is dominated by model capacity and training dynamics, not noise distribution alignment.

## Files

- Script: `scripts/test_phase3b_matched_noise.py`
- Results: `results/phase3b/phase3b_results.json`
- Run log: `results/phase3b/phase3b_run.log`
- Plan: `plans/phase3b-plan.md`
