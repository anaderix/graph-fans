# Phase 5a Run Log

## Execution Environment
- **GPU:** NVIDIA L40S (46GB VRAM), Nebius Cloud VM `l40s-intel-vm`
- **Instance ID:** computeinstance-e00gtkekzp8dhy82w7
- **Date:** 2026-04-15

## Command
```bash
uv run python scripts/test_phase5a.py \
    --families "SBM(q=0.05),SBM(q=0.1),BA(m=2),BA(m=5)" \
    --n-nodes 50 --n-seeds 5 --n-epochs 500 \
    --device cuda --output results/phase5a/phase5a_results.json \
    --dataset-dir results/phase4a/datasets
```

## Runtime
~45 minutes total (spatial baselines ~95-110s each, spectral ~7s each)

## Results

| Family | Uniform W1 | Band W1 | Spectral W1 | Spec vs Band |
|--------|-----------|---------|-------------|-------------|
| SBM(q=0.05) | 714.5 | 627.9 | 379,890.8 | -60,400% |
| SBM(q=0.1) | 1,036.0 | 963.7 | 373,893.6 | -38,700% |
| BA(m=2) | 675.9 | 618.6 | 367,801.4 | -59,354% |
| BA(m=5) | 1,380.1 | 1,308.6 | 389,724.6 | -29,682% |

All spectral conditions significantly worse (p < 0.0001).
Band vs uniform replicates Phase 2g across all 4 families.

### Training Diagnostics
- Spectral MLP final loss: 0.957-0.986 (barely below 1.0 = random prediction)
- Spatial GCN final loss: 0.33-0.44 (well-trained)
- Spectral training time: 1.2s/run (vs 95-110s for spatial)
- gen_neighbor_correlation: 1.000 for all spectral runs (degenerate: all nodes get identical features)

## Gate Decision
**Phase 5a: NO-GO** — Independent per-mode spectral diffusion fails. MLP does not learn per-mode denoising.

## Root Cause
The MLP loss stays ~1.0 across all 500 epochs, indicating it never learns to denoise. Two likely factors:
1. **Insufficient gradient steps per mode:** 500 epochs with ~100 samples = ~50,000 MLP forward passes across all modes, but each mode only sees ~1000 updates. The spatial GCN gets 16,000 gradient steps on correlated full-graph features.
2. **Mode collapse in generation:** gen_neighbor=1.000 suggests generated features are constant across nodes (all modes collapse to near-zero coefficients or DC component). The per-mode DDIM produces degenerate trajectories when the MLP predicts near-zero noise (loss ~1.0).

## Artifacts
- `results/phase5a/phase5a_results.json`
