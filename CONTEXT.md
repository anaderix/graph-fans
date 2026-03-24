# Graph-FANS: Project Context for New Agent

## What This Project Is

Graph-FANS applies spectral-aware noise shaping from FANS (image diffusion, ICLR 2026) to graph diffusion models. The core idea: shape diffusion noise in the Laplacian eigenbasis to preserve multi-scale spectral structure during generation.

**Vault location:** `~/docker-stacks/obsidian/config/Desktop/vault/1-Project/2026-GraphFANS/`
**Code location:** `~/projects/graph-fans/` (this repo)
**Remote GPU:** `ssh -A anaderi@89.169.122.217` (NVIDIA L40S, project at same path)
**GitHub:** `git@github.com:anaderix/graph-fans.git`

## Project Structure

```
graph_fans/
├── utils/
│   ├── graph_generators.py     # SBM, BA, citation network generators
│   ├── multiscale_features.py  # Feature generation: smooth, multiscale, community modes
│   └── save.py                 # save_figure(fig, path, title) → PNG + companion .md
├── phase0/
│   ├── spectral_profiler.py    # Laplacian eigendecomp, band partitioning, energy profiling
│   ├── run_profiling.py        # Main Phase 0 script
│   └── visualize.py
├── phase1a/
│   ├── spectral_metrics.py     # JSD, QBE, HKS metrics + MMD baselines
│   ├── metric_validator.py     # G1 gate validation framework
│   ├── run_validation.py
│   └── visualize.py
├── phase2/
│   ├── noise_shaper.py         # FANS importance weights + noise shaping in eigenbasis
│   ├── score_network.py        # 3-layer GCN with sinusoidal time embedding
│   ├── sde.py                  # VPSDE (linear beta) + CosineScheduleSDE
│   ├── spectral_loss.py        # Tweedie denoising + per-band spectral fidelity loss
│   ├── trainer.py              # Training loop + generation (NEEDS FIX - see below)
│   ├── evaluate.py             # H1-A and H2 experiment pipelines + G2 gate
│   ├── run_experiment.py       # CLI entry point
│   └── visualize.py
tests/
├── test_phase0.py              # 15 tests
├── test_phase1a.py             # 13 tests
└── test_phase2.py              # 18 tests (some test broken behavior)
plans/
├── phase2-plan.md              # Original execution plan
└── phase2-review.md            # Code review findings
results/
├── phase0/                     # G0: GO (9/9 families, energy ratios 3.8×–17×)
├── phase1a/                    # G1: GO (QBE reveals gaps invisible to MMD)
├── phase2/                     # 2a: NO-GO (broken training, SBM -120%)
├── phase2b/                    # 2b: NO-GO (cosine+EMA fixed stability, still no-op)
├── phase2c/                    # 2c: NO-GO (spectral loss had zero effect)
├── phase2d/                    # 2d: NO-GO (multiscale features, still no-op)
├── phase2e/                    # 2e: NO-GO (500-sample dataset, still no-op)
└── graph_diagnostics/          # Feature mode comparison visualizations
scripts/
├── build_reports.py            # HTML report generator
└── visualize_synthetic_graphs.py
```

## Current State: Phase 2 Training Is Broken

**ALL Phase 2 results (2a–2e) are invalid.** The score network generates pure noise.

### Confirmed Root Causes

1. **Wrong training target** (`trainer.py:187-191`):
   - Code has `target = -noise / std` (score formulation)
   - At small std (near t=0), target explodes → unstable training
   - **Fix:** Switch to ε-prediction: `target = noise`

2. **Reverse SDE adds noise at t→0** (`sde.py` reverse_step + `trainer.py` generate):
   - Every reverse step adds `diffusion * sqrt(|dt|) * noise`, even at the final step
   - Final output is corrupted by accumulated stochastic noise
   - **Fix:** Use DDIM deterministic reverse steps, Tweedie formula at t=0

3. **Insufficient training** (`trainer.py` TrainConfig):
   - Default: 500 epochs × 16 timesteps = 8,000 steps for 500 samples = 16 passes/sample
   - **Fix:** Increase to 2000 epochs × 32 timesteps = 64,000 steps = 128 passes/sample

4. **No dataset persistence** — features generated on-the-fly, can't inspect before training
   - **Fix:** New `dataset.py` module to pre-generate, save as .npz, and cache

### Evidence of Failure

```
ref profile:   [0.386  0.  0.376  0.01  0.058  0.064  0.065  0.041]  ← bimodal (community)
gen profile:   [0.054  0.  0.016  0.035  0.217  0.258  0.246  0.173]  ← matches random noise
noise profile: [0.003  0.  0.015  0.034  0.223  0.263  0.280  0.182]  ← pure random
gen std: 13.7, ref std: 1.3  ← 10× too high
```

The QBE metric was constant across all experiments because it measured noise-vs-noise distance.

## The Plan (Ready to Execute)

Full plan at `~/.claude/plans/zany-churning-dolphin.md`. Summary:

1. **`sde.py`** — Add `alpha_bar()` + `ddim_step()` to both SDE classes
2. **`spectral_loss.py`** — Fix Tweedie: `x_hat_0 = (x_t - std * eps_pred) / mean_coeff`
3. **`dataset.py`** (NEW) — Pre-generate, save .npz, load, validate feature datasets
4. **`trainer.py`** — Fix target to `noise`, replace generation with DDIM, update defaults, add sanity check
5. **`evaluate.py`** — Use cached datasets, add pre-generation loop, add sanity check
6. **`run_experiment.py`** — Add `--dataset-dir`, `--pre-generate-only`, update defaults
7. **`tests/test_phase2.py`** — Update + new tests for ε-prediction, DDIM, dataset caching

## Key Design Decisions Already Made

- **ε-prediction** (not score prediction) — simpler, standard DDPM formulation
- **DDIM deterministic sampling** (eta=0) — eliminates noise-at-t=0 problem
- **Cosine noise schedule** — proven stable in 2b (vs linear VP-SDE instability in 2a)
- **EMA** (decay=0.999) — proven helpful in 2b
- **Community features** — bimodal spectral profile (72× energy ratio), role-dependent
- **500 training samples per graph** — proper distributional training (not single-sample memorization)

## What Worked (Phases 0 and 1a)

- **Phase 0 (G0: GO):** Graph spectral energy IS non-uniform (3.8×–17× ratios). The premise holds.
- **Phase 1a (G1: GO):** QBE metric reveals spectral fidelity gaps invisible to standard MMD (up to 63×). The evaluation tool works.
- These results are valid and don't need to be re-run.

## Vault Reports

- `1-Project/2026-GraphFANS/Report-Phase0.md` — Spectral profiling results
- `1-Project/2026-GraphFANS/Report-Phase1a.md` — Metrics validation results
- `1-Project/2026-GraphFANS/Report-Phase2.md` — Consolidated negative result (5 iterations)
- `1-Project/2026-GraphFANS/Note-Synthetic-Graph-Diagnostics.md` — Feature mode analysis
- `1-Project/2026-GraphFANS/Phase2-Experiment-Diagrams.md` — Mermaid data flow diagrams
- `1-Project/2026-GraphFANS/backlog.md` — 11 backlog items from code review
- `1-Project/2026-GraphFANS/Roadmap-Hypotheses-Validation.md` — Full research roadmap

## Running Experiments

```bash
# Local (CPU, for testing)
cd ~/projects/graph-fans
uv run pytest tests/ -v
uv run python -m graph_fans.phase2 --epochs 50 --seeds 1 --n-nodes 30

# Remote GPU
ssh -A anaderi@89.169.122.217
cd projects/graph-fans && git pull && source ~/.local/bin/env
uv run python -m graph_fans.phase2 --device cuda --epochs 2000 --seeds 5
```
