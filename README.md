# Graph-FANS

**Frequency-Adaptive Noise Shaping on Graph Laplacian Spectra**

Applies spectral-aware noise shaping from [FANS](https://arxiv.org/abs/2410.02204) (ICLR 2026) to graph diffusion models. The core idea: shape diffusion noise in the Laplacian eigenbasis to preserve multi-scale spectral structure during generation.

## Results

| Phase | Gate | Finding |
|-------|------|---------|
| Phase 0 | G0: GO | Graph spectral energy is non-uniform (3.8-17x ratios across 9 families) |
| Phase 1a | G1: GO | QBE metric reveals spectral fidelity gaps invisible to standard MMD (up to 63x) |
| Phase 2a-2e | NO-GO | 5 iterations failed due to broken training (score formulation, stochastic sampling, insufficient epochs) |
| Phase 2f | NO-GO | Fixed training (epsilon-prediction + DDIM), but model still generates noise at 200 nodes |
| **Phase 2g** | **CONDITIONAL GO** | **Spectral shaping improves W1 by 6-12% on 3/4 families (p<0.01) at n=50 with 3L GCN** |

The spectral shaping effect is real but conditional: requires a model capable of actual denoising (n=50), a distributional metric (W1, not QBE), and graphs with multi-scale spectral structure.

## Setup

```bash
uv sync
uv run pytest tests/ -v
```

## Usage

```bash
# Run spectral shaping experiment
uv run python scripts/test_shaping_w1.py --families "SBM(q=0.05),BA(m=2)" --n-seeds 5 --device cpu

# Run Phase 2 training pipeline
uv run python -m graph_fans.phase2 --epochs 500 --seeds 3 --n-nodes 50
```

## Project structure

```
graph_fans/
  phase0/          Spectral profiling
  phase1a/         Spectral metrics (JSD, QBE, HKS)
  phase2/          Diffusion training + evaluation
  utils/           Graph generators, feature generation
tests/             78 tests
scripts/           Experiment scripts
plans/             Experiment plans and roadmap
results/           Outputs by phase
```

See `LOG.md` for the full experiment log and `CLAUDE.md` for development conventions.
