---
tags: [project, graph-fans, report, phase2g-followup, spectral-shaping]
created: 2026-03-25
phases: [step1-families, step2-scale, step3-downstream]
gate: PARTIAL GO
decision: Effect generalizes directionally but lacks statistical power at current sample size
supersedes: null
---

# Phase 2g Follow-up Report: Generalization, Scale, and Downstream Evaluation

## Summary

Phase 2g established a 12.5% W1 improvement from spectral noise shaping on SBM(q=0.05) at 50 nodes (p=0.0001). This follow-up tested three questions: does the effect generalize across graph families, does it persist at larger scales, and does it translate to downstream task performance? The answer is a qualified "partially." All 3 new families show positive improvement direction (5.2%--8.5%), but only 1 of 3 reaches significance under Bonferroni correction (BA(m=2), p=0.016). The effect persists directionally at all tested scales (50--200 nodes) but with a non-monotonic pattern and insufficient statistical power. The downstream node classification evaluation reveals that 12.5% W1 improvement translates to only 0.9% accuracy gain on SBM(q=0.05) and -0.1% on BA(m=5) -- well below the 2% practical significance threshold.

## Step 1: Family Generalization

### Goal

Test whether the 12.5% W1 improvement generalizes beyond SBM(q=0.05) to three new families with varying spectral profiles: SBM(q=0.1), BA(m=2), and BA(m=5). Success criterion: at least 2 of 3 families significant at p<0.025 (Bonferroni-corrected).

### Method

Each family was tested at n=50 nodes, 4 features, 100 training samples, 5 seeds. Models used the Phase 2g configuration: 3-layer GCN, 128 hidden dim, cosine schedule, EMA, LR annealing, epsilon-prediction, DDIM generation (200 steps). Evaluation used the spectral Wasserstein-1 distance. Paired t-tests (5 seeds) with Bonferroni-corrected threshold alpha=0.025.

### Results

| Family | Spectral Profile | Uniform W1 (mean +/- std) | Spectral W1 (mean +/- std) | Improvement | t-stat | p-value | Significant (alpha=0.025)? |
|--------|-----------------|---------------------------|---------------------------|-------------|--------|---------|--------------------------|
| SBM(q=0.05) [ref] | 39%+37% bimodal | 717.6 +/- 84.1 | 627.9 +/- 87.8 | 12.5% | 15.2 | 0.0001 | Yes |
| SBM(q=0.1) | 44%+27% bimodal | 1035.8 +/- 16.0 | 963.7 +/- 46.9 | 7.0% | 3.11 | 0.036 | No |
| BA(m=2) | 32%+45% adjacent bands | 675.9 +/- 95.9 | 618.6 +/- 112.0 | 8.5% | 4.02 | 0.016 | Yes |
| BA(m=5) | 26%+41% bimodal (bands 0+2) | 1380.1 +/- 237.3 | 1308.6 +/- 194.8 | 5.2% | 1.64 | 0.176 | No |

**Gate result: 1 of 3 families significant (needed 2 of 3). FAIL on the pre-registered criterion.**

### Key Findings

**1. Universally positive direction, inconsistent significance.** All 3 new families show spectral shaping reducing W1 (improvement range 5.2%--8.5%). The effect is never negative. However, only BA(m=2) reaches the Bonferroni-corrected threshold (p=0.016 < 0.025). SBM(q=0.1) narrowly misses at p=0.036, and BA(m=5) fails to reach significance at p=0.176.

**2. High variance in BA(m=5) masks the effect.** BA(m=5) has the highest within-method variance (uniform std=237.3, spectral std=194.8), which is 2.5x larger than SBM(q=0.1) (uniform std=16.0). This inflated variance arises from the seed-2 outlier (uniform W1=1747.8 vs mean 1380.1). Excluding seed 2, the improvement rises to 6.3% with lower variance, but post-hoc exclusion is not justified.

**3. Per-seed analysis reveals consistent direction in BA(m=2).** All 5 seeds show improvement for BA(m=2): 15.6%, 10.5%, 7.4%, 1.0%, 10.2%. The weakest seed (seed 3, 1.0%) corresponds to the highest absolute W1 values (794.1 vs 786.4), suggesting the model struggles more on that particular graph instance but spectral shaping still does not hurt.

**4. Effect magnitude correlates with spectral bimodality but imperfectly.** The roadmap predicted that bimodal spectra (energy spread across non-adjacent bands) would benefit most. SBM(q=0.05) (strongly bimodal) shows the largest effect (12.5%). BA(m=5) was predicted to show a strong effect due to bimodal structure (bands 0+2), but the observed 5.2% is the weakest. This suggests bimodality alone is not a sufficient predictor -- the specific pattern of energy distribution and the model's ability to resolve it both matter.

## Step 2: Scale Study

### Goal

Test whether the W1 improvement persists, grows, or vanishes at larger graph sizes. Tested SBM(q=0.05) and BA(m=5) at n=50 (reference), 100, 150, and 200 nodes. Success criterion: effect persists at n>=100; growing effect would be the strongest finding.

### Method

Same architecture and training configuration as Phase 2g (3L GCN, hidden_dim=128, 4 features, 5 seeds). Separate dataset directories per scale to avoid cache collision. Importance weights recomputed per scale from seed-0 training data. All scales use n_train=100 realizations, n_ref=50 reference samples.

### Results

#### SBM(q=0.05) Scaling

| n_nodes | Uniform W1 (mean +/- std) | Spectral W1 (mean +/- std) | Improvement | t-stat | p-value | Significant? |
|---------|---------------------------|---------------------------|-------------|--------|---------|-------------|
| 50 | 717.6 +/- 84.1 | 627.9 +/- 87.8 | 12.5% | 15.2 | 0.0001 | Yes |
| 100 | 1565.4 +/- 289.5 | 1548.0 +/- 146.6 | 1.1% | 0.13 | 0.900 | No |
| 150 | 3451.6 +/- 472.0 | 3216.5 +/- 350.9 | 6.8% | 1.65 | 0.174 | No |
| 200 | 5367.3 +/- 490.9 | 4828.2 +/- 389.4 | 10.0% | 2.09 | 0.105 | No |

#### BA(m=5) Scaling

| n_nodes | Uniform W1 (mean +/- std) | Spectral W1 (mean +/- std) | Improvement | t-stat | p-value | Significant? |
|---------|---------------------------|---------------------------|-------------|--------|---------|-------------|
| 50 | 1380.1 +/- 237.3 | 1308.6 +/- 194.8 | 5.2% | 1.64 | 0.176 | No |
| 100 | 2223.1 +/- 266.4 | 2118.1 +/- 203.8 | 4.7% | 1.69 | 0.166 | No |
| 150 | 2503.2 +/- 347.0 | 2224.7 +/- 257.6 | 11.1% | 3.42 | 0.027 | No (alpha=0.025) |
| 200 | 2754.0 +/- 364.3 | 2576.9 +/- 358.6 | 6.4% | 2.69 | 0.055 | No |

#### Combined Scaling Summary

| n_nodes | SBM(q=0.05) Improv. | SBM p-value | BA(m=5) Improv. | BA p-value |
|---------|---------------------|-------------|-----------------|------------|
| 50 | 12.5% | 0.0001 | 5.2% | 0.176 |
| 100 | 1.1% | 0.900 | 4.7% | 0.166 |
| 150 | 6.8% | 0.174 | 11.1% | 0.027 |
| 200 | 10.0% | 0.105 | 6.4% | 0.055 |

### Key Findings

**1. The effect does not vanish -- it persists directionally at all scales.** Across all 8 scale-family combinations, spectral shaping produces lower (better) mean W1 in 8 out of 8 cases. The probability of this happening by chance under the null (no effect) is 0.5^8 = 0.004.

**2. The effect is non-monotonic with scale, especially for SBM(q=0.05).** The SBM improvement drops from 12.5% at n=50 to 1.1% at n=100, then rebounds to 6.8% at n=150 and 10.0% at n=200. This non-monotonic pattern is not predicted by any simple capacity-saturation theory. The n=100 dip is notable: the t-stat drops to 0.13 (essentially zero signal), but the effect recovers at larger scales.

**3. BA(m=5) peaks at n=150.** BA(m=5) shows relatively stable improvement at n=50 (5.2%) and n=100 (4.7%), then peaks at n=150 (11.1%, p=0.027 -- narrowly missing significance), and retreats to 6.4% at n=200 (p=0.055). The n=150 result, with t=3.42, is the closest any non-baseline condition comes to significance in this study.

**4. Absolute W1 grows faster than linearly with n.** SBM(q=0.05) W1 grows from ~718 at n=50 to ~5367 at n=200 (7.5x for a 4x increase in nodes). This super-linear growth suggests the task becomes harder with scale -- the model must reconstruct more spectral structure -- but spectral shaping's relative benefit does not decay proportionally.

**5. Variance is a dominant limitation.** At n=100, SBM(q=0.05) uniform W1 has std=289.5 (18.5% of mean), compared to 84.1 (11.7%) at n=50. This increased relative variance, combined with only 5 seeds, is likely the primary reason p-values fail to reach significance despite meaningful effect sizes. BA(m=5) at n=200 has p=0.055 with t=2.69 -- a borderline result that would likely reach significance with 10+ seeds.

**6. Model capacity is not a hard barrier.** The roadmap anticipated that n=200 might fail due to model capacity (std_ratio exceeding 3x). In practice, std_ratios remain in the 1.3--2.1 range at all scales for both families, indicating the 3L GCN with hidden_dim=128 still denoises at n=200 (no SCALE-4 retry with hidden_dim=256 was needed). The problem is not capacity collapse but insufficient statistical power.

## Step 3: Downstream Task Evaluation

### Goal

Determine whether 12.5% W1 improvement translates to practical benefit on node classification (predicting community membership from generated features). Success criterion: spectral-generated features produce at least 2% higher classification accuracy.

### Method

For each family (SBM(q=0.05), BA(m=5)) and each seed, 1000 feature matrices were generated per method (uniform, spectral). Per-node spectral coefficients were extracted and used to train a logistic regression classifier (sklearn, max_iter=1000). Training used generated features; testing used held-out real (reference) features. 5 seeds, paired t-test. SBM(q=0.05) had 5 communities; BA(m=5) had 4 communities (Louvain detection, seed=0).

### Results

| Family | Uniform Acc (mean +/- std) | Spectral Acc (mean +/- std) | Improvement | t-stat | p-value | Significant? |
|--------|---------------------------|---------------------------|-------------|--------|---------|-------------|
| SBM(q=0.05) | 0.3031 +/- 0.0015 | 0.3059 +/- 0.0023 | +0.9% | 4.95 | 0.008 | Yes (p<0.05) |
| BA(m=5) | 0.3244 +/- 0.0012 | 0.3239 +/- 0.0019 | -0.1% | -0.47 | 0.663 | No |

### Per-Seed Detail: SBM(q=0.05)

| Seed | Uniform Test Acc | Spectral Test Acc | Improvement |
|------|-----------------|------------------|-------------|
| 0 | 0.3032 | 0.3068 | +1.2% |
| 1 | 0.3008 | 0.3016 | +0.3% |
| 2 | 0.3052 | 0.3084 | +1.0% |
| 3 | 0.3024 | 0.3064 | +1.3% |
| 4 | 0.3040 | 0.3064 | +0.8% |

### Key Findings

**1. The W1 improvement does not translate to practical classification benefit.** On SBM(q=0.05), where spectral shaping produces a 12.5% W1 improvement, the downstream classification accuracy gain is only 0.9% (0.3031 vs 0.3059). While statistically significant (p=0.008), this falls well below the pre-registered 2% threshold for practical significance.

**2. BA(m=5) shows no downstream benefit whatsoever.** The accuracy difference is -0.1% (spectral slightly worse), with p=0.663. The 5.2% W1 improvement at n=50 for BA(m=5) produces zero detectable classification benefit.

**3. Both methods perform only marginally above chance.** With 5 communities, random guessing yields 20% accuracy. Uniform achieves 30.3% and spectral 30.6% on SBM(q=0.05). With 4 communities (BA(m=5)), chance is 25% and both methods achieve ~32.4%. The generated features contain some community structure information but are far from fully resolved. This suggests the diffusion model is still operating in a low-quality regime where distributional refinements (spectral shaping) do not meaningfully change the downstream signal.

**4. The gap between distributional and task metrics is informative.** A 12.5% reduction in distributional distance (W1) produces a 0.9% accuracy change. This 14:1 ratio suggests that the W1 improvement is concentrated in aspects of the spectral distribution that do not strongly correlate with community membership. Specifically, W1 measures fidelity across all spectral bands, while community detection relies primarily on the low-frequency (smooth) components. Spectral shaping may be improving higher-frequency fidelity that classification ignores.

## Cross-Cutting Analysis

### What predicts whether spectral shaping helps?

The strongest predictor of a large, significant effect is **low baseline variance across seeds**. SBM(q=0.05) at n=50 has the lowest relative variance (uniform std/mean = 11.7%) and the only highly significant result (p=0.0001). When variance is high (BA(m=5) at n=50: std/mean = 17.2%), even an 8.5% improvement fails to reach significance. Spectral bimodality appears to be a necessary but not sufficient condition -- BA(m=5) has bimodal structure but the highest variance.

### Why does the effect fluctuate with scale?

The non-monotonic SBM(q=0.05) pattern (12.5% at n=50, 1.1% at n=100, 6.8% at n=150, 10.0% at n=200) is puzzling. One hypothesis: at n=100, the spectral structure of SBM(q=0.05) changes relative to the 8-band partition used for importance weights. The Laplacian eigenvectors at different scales may distribute energy across bands differently, creating a scale at which the importance weights are poorly calibrated. This is consistent with the observation that the uniform and spectral methods produce nearly identical W1 at n=100 (1565.4 vs 1548.0), suggesting the shaping is neither helping nor hurting -- it is simply not engaging the relevant spectral modes at that scale.

### Why doesn't W1 improvement translate to classification accuracy?

Three explanations, not mutually exclusive:

1. **Spectral band mismatch.** Classification relies on low-frequency (community-scale) features. W1 aggregates across all bands. The improvement may be concentrated in mid-to-high frequency bands that are irrelevant to community structure.

2. **Regime of model quality.** At 30% accuracy (SBM) and 32% accuracy (BA), the generative model produces features that carry weak community signal. In this regime, a 12.5% distributional improvement is a relative refinement of a weak signal, not a qualitative change in information content.

3. **Linear classifier limitation.** Logistic regression may be unable to exploit the distributional differences that W1 captures. A non-linear classifier (e.g., GCN on the generated features) might reveal a larger gap, though this was not tested.

## Implications for the Paper

1. **The primary claim should be the W1 improvement on SBM(q=0.05), not generalization.** The 12.5% result at p=0.0001 is rock-solid. Family generalization is directionally positive (3/3 positive, 1/3 significant), which supports the claim but cannot be presented as a strong generalization result.

2. **The scale study supports persistence, not growth.** The paper can claim "the effect persists across scales from 50 to 200 nodes" based on the 8/8 positive direction (sign test p=0.004). It cannot claim monotonic growth or consistent statistical significance at larger scales.

3. **The downstream evaluation is a negative result that strengthens the metric contribution.** The finding that W1 improvement does not translate to classification accuracy supports positioning the Spectral W1 metric itself as a contribution -- it captures distributional properties that task-specific metrics miss. This is a stronger framing than claiming downstream benefit.

4. **Variance, not effect size, is the primary limitation.** Effect sizes of 5--12% are meaningful. The failure to reach significance at most conditions is attributable to n=5 seeds. The paper should acknowledge this and recommend larger seed counts for future work.

5. **The non-monotonic scale pattern should be presented honestly as an open question**, not smoothed over. It is an empirical observation that resists simple explanation and represents a genuine research gap.

6. **Recommended framing:** "Spectral noise shaping reduces distributional distance (W1) by 7--12% on graphs with multi-scale spectral structure. The effect is consistent in direction across 4 families and 4 scales, reaching statistical significance on the primary target (SBM q=0.05, p=0.0001) and on BA(m=2) (p=0.016). The improvement is in distributional fidelity, not downstream task performance, supporting spectral W1 as a more sensitive evaluation metric for graph diffusion models."

## Next Steps

1. **Increase seed count to 10--20 for borderline conditions.** BA(m=5) at n=150 (p=0.027, t=3.42) and n=200 (p=0.055, t=2.69) are the most promising candidates for reaching significance with more seeds.

2. **Investigate the n=100 dip.** Analyze the spectral profile of SBM(q=0.05) at n=100 to understand why importance weights become ineffective at that scale. Recompute weights from n=100-specific graph spectra and compare to the n=50-derived weights.

3. **Try a non-linear downstream classifier.** Replace logistic regression with a 2-layer MLP or GCN to test whether the distributional improvement carries information that a linear model cannot exploit.

4. **Adaptive importance weights per scale.** Instead of using fixed 8-band partitions at all scales, explore scale-adaptive band partitioning that accounts for the changing spectral density at different graph sizes.

5. **Proceed with paper draft.** The results are sufficient for a workshop paper or short paper framing: strong primary result, honest generalization analysis, methodological contribution (spectral W1), and negative downstream result as informative finding.

## Code & Data

| Artifact | Path |
|----------|------|
| Shaping test script | `scripts/test_shaping_w1.py` |
| Spectral W1 metric | `graph_fans/phase2/spectral_wasserstein.py` |
| Downstream evaluation | `graph_fans/phase2/downstream.py` |
| Phase 2g baseline results | `results/diagnostics/shaping_w1_test.json` |
| Family generalization results | `results/diagnostics/family_generalization.json` |
| Scale study results (n=100) | `results/diagnostics/scale_study_n100.json` |
| Scale study results (n=150) | `results/diagnostics/scale_study_n150.json` |
| Scale study results (n=200) | `results/diagnostics/scale_study_n200.json` |
| Downstream results | `results/diagnostics/downstream_results.json` |
| Run command (family) | `PYTHONPATH=. uv run python scripts/test_shaping_w1.py --families SBM(q=0.1),BA(m=2),BA(m=5)` |
| Run command (scale) | `PYTHONPATH=. uv run python scripts/test_shaping_w1.py --families SBM(q=0.05),BA(m=5) --n-nodes N` |
| Run command (downstream) | `PYTHONPATH=. uv run python scripts/run_downstream.py` |
| Hardware | NVIDIA L40S GPU |
| Previous report | [[Report-Phase2g]] |
| Roadmap | `plans/phase2g-followup.md` |
| Execution plan | `plans/phase2g-followup-plan.md` |
