# Phase 4a + 4b Run Log

## Execution Environment
- **GPU:** NVIDIA L40S (46GB VRAM), Nebius Cloud VM `l40s-intel-vm`
- **Instance ID:** computeinstance-e00gtkekzp8dhy82w7
- **Date:** 2026-04-15

## Phase 4a: InfoNoise-Guided Training

### Command
```bash
uv run python scripts/test_info_noise.py \
    --families "SBM(q=0.05),BA(m=2)" \
    --n-nodes 50 --n-seeds 5 --device cuda \
    --output results/phase4a/info_noise_results.json \
    --dataset-dir results/phase4a/datasets \
    --n-epochs 500 --save-profiles
```

### Runtime
~60 minutes (40 training runs)

### Results

| Family | uniform | band | info_noise | info_noise+band |
|--------|---------|------|------------|-----------------|
| SBM(q=0.05) | 728.6 | 627.9 | 4,201,214.6 | 4,195,543.2 |
| BA(m=2) | 675.9 | 618.6 | 2,105,523.1 | 2,156,509.1 |

- Band vs uniform: SBM 13.8% (p=0.0011), BA 8.5% (p=0.0158) — replicates Phase 2g
- InfoNoise vs uniform: SBM -576,488% (p<0.0001), BA -311,424% (p<0.0001) — catastrophic failure
- InfoNoise models produce pure noise: std_ratio 85-100x, sanity checks fail on all seeds

### Gate Decision
**Phase 4a: NO-GO** — InfoNoise adaptive t-sampling catastrophically fails for graph diffusion.

### Root Cause
Entropy rate r(σ) = mmse(σ)/σ³ dominated by 1/σ³ divergence. GCN loss ≈ 1.0 at low sigma (capacity ceiling), so entropy rate concentrates 80% of sampling at σ < 0.08. Death spiral: model only trains where it can't learn.

### Artifacts
- `results/phase4a/info_noise_results.json`
- `results/phase4a/entropy_rate_profiles/` (20 profile JSONs)
- `results/phase4a/datasets/` (cached datasets)

---

## Phase 4b: InfoGrid for DDIM

### Command
```bash
uv run python scripts/test_info_grid.py \
    --families "SBM(q=0.05),BA(m=2)" \
    --n-seeds 5 --device cuda \
    --output results/phase4b/info_grid_results.json \
    --dataset-dir results/phase4a/datasets \
    --profile-dir results/phase4a/entropy_rate_profiles \
    --step-budgets "200,100,50"
```

### Runtime
~50 minutes (20 training runs + 120 generation runs)

### Results

| Config | Uniform Grid W1 | InfoGrid W1 | Change |
|--------|-----------------|-------------|--------|
| SBM/uniform/200 | 727.9 | 34,337.6 | -4617% |
| SBM/uniform/100 | 876.1 | 45,052.6 | -5043% |
| SBM/uniform/50 | 1,119.9 | 68,061.9 | -5977% |
| SBM/band/200 | 627.9 | 33,797.9 | -5283% |
| SBM/band/100 | 749.8 | 44,629.6 | -5852% |
| SBM/band/50 | 982.9 | 67,330.8 | -6750% |
| BA/uniform/200 | 675.9 | 15,634.9 | -2213% |
| BA/uniform/100 | 837.8 | 21,157.5 | -2425% |
| BA/uniform/50 | 1,066.0 | 33,167.1 | -3012% |
| BA/band/200 | 618.6 | 14,908.9 | -2310% |
| BA/band/100 | 778.3 | 20,355.7 | -2515% |
| BA/band/50 | 1,042.9 | 32,302.8 | -2998% |

All InfoGrid conditions significantly worse (p < 0.0001).

### Gate Decision
**Phase 4b: NO-GO** — InfoGrid concentrates DDIM steps at low sigma where model loss ≈ 1.0, corrupting output.

### Root Cause
Same as 4a: entropy rate profile from uniform-trained models still has 1/σ³ divergence at low sigma. InfoGrid puts more DDIM steps where the model does harmful denoising (loss ~1.0 = predicting zero), taking steps away from mid-high sigma where the model actually works.

### Artifacts
- `results/phase4b/info_grid_results.json`

---

## Overall Summary

Both Phase 4a and 4b are NO-GO. The fundamental issue: InfoNoise assumes the model is reasonably close to Bayes-optimal across all noise levels. The 3L GCN on graph features has a hard capacity ceiling at low noise (loss ~1.0, confirmed in Phase 2f NS-D diagnostic). The entropy rate formula's 1/σ³ factor amplifies this regime, concentrating both training samples (4a) and DDIM steps (4b) where the model is worst.

Band-mismatched spectral shaping from Phase 2g remains the best approach.

### Positive Finding (Diagnostic)
The entropy rate profile for graph feature diffusion is a novel diagnostic that reveals:
1. The informative window concept from InfoNoise does not transfer to capacity-limited models
2. The 3L GCN's loss profile (flat ~1.0 at low sigma) is qualitatively different from image denoising models
3. This explains why the NS-A/NS-C gradient rebalancing also failed (Phase 2f): the problem is model capacity at low noise, not training allocation
