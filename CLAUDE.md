# Graph-FANS

Spectral-aware noise shaping from FANS (ICLR 2026) applied to graph diffusion models. Shapes diffusion noise in the Laplacian eigenbasis to preserve multi-scale spectral structure during graph feature generation.

## Quick start

```bash
uv run pytest tests/ -v          # 78 tests
uv run python -m graph_fans.phase2 --epochs 50 --seeds 1 --n-nodes 30  # quick local run
```

## Project structure

- `graph_fans/phase0/` — Spectral profiling (eigendecomp, band partitioning, energy profiles)
- `graph_fans/phase1a/` — Spectral metrics (JSD, QBE, HKS) and metric validation
- `graph_fans/phase2/` — Diffusion training: noise shaping, score network (GCN), SDE, trainer, evaluation
- `graph_fans/utils/` — Graph generators (SBM, BA, citation), feature generation (smooth/multiscale/community)
- `tests/` — Unit tests for all phases
- `scripts/` — Experiment scripts (standalone, not imported by the package)
- `plans/` — Experiment plans and roadmap
- `results/` — Outputs organized by phase (phase0/, phase1a/, phase2/, phase2b-2f/, phase3a/, diagnostics/, graph_diagnostics/)

## Key modules

| Module | Purpose |
|--------|---------|
| `phase2/noise_shaper.py` | FANS importance weights + noise shaping in eigenbasis |
| `phase2/trainer.py` | Training loop (epsilon-prediction, DDIM generation, EMA) |
| `phase2/sde.py` | VPSDE + CosineScheduleSDE with alpha_bar/ddim_step |
| `phase2/spectral_wasserstein.py` | Per-mode W1 metric (primary evaluation metric since Phase 2f) |
| `phase2/evaluate.py` | H1-A experiment pipeline + G2 gate |
| `phase2/dataset.py` | Feature dataset generation, caching (.npz), validation |
| `phase0/spectral_profiler.py` | Laplacian eigendecomp, band partitioning, band energy |
| `utils/multiscale_features.py` | Community/multiscale/smooth feature generation |
| `utils/graph_generators.py` | SBM, BA, citation network generators |

## Conventions

- **Evaluation metric:** Spectral Wasserstein-1 (per-eigenmode distributional distance). QBE was used in early phases but is too coarse — W1 is the standard from Phase 2f onward.
- **Training:** Epsilon-prediction + DDIM deterministic sampling + cosine SDE + EMA (0.999) + LR annealing. These were validated in Phase 2b-2f.
- **Features:** Community mode (`generate_community_boundary_features`) for experiments. Bimodal spectral profile gives shaping the most to work with.
- **Statistical tests:** Paired t-test with Bonferroni correction. Alpha adjusted by number of families.
- **Dataset splits:** Train and ref splits use separate base seeds. Importance weights derived from seed-0 training split only (no data leakage to ref).
- **Results:** Each phase writes to its own `results/` subdirectory. JSON for structured data, PNG+MD for figures.

## Current state

- **G0: GO** — Graph spectra are non-uniform (3.8-17x energy ratios)
- **G1: GO** — QBE metric reveals gaps invisible to MMD
- **G2: CONDITIONAL GO** — Spectral shaping works at n=50 with W1 metric (6-12% improvement, significant across 3/4 families after power boost)
- **H2 (temporal ramp): DEAD** — Removed from codebase, <1% effect
- **Phase 3 (persistence bands): DEPRIORITIZED** — No meaningful scale separation in persistence at n=50
- **Phase 3a (per-mode shaping): NO-GO** — Per-mode shaping worsens W1 by 50-94% due to train/generate noise mismatch. Band shaping (Phase 2g) remains best.
- **Phase 3b (matched gen noise): NO-GO** — Matching generation start to shaped training noise worsens W1. DDIM compensates for the mismatch; band-mismatched (Phase 2g) remains best.

## Running experiments

```bash
# Phase 2g-style shaping test (W1 metric)
uv run python scripts/test_shaping_w1.py --families "SBM(q=0.05)" --n-seeds 5 --device cpu

# Phase 3a (per-mode shaping, NO-GO — results only)
uv run python scripts/test_phase3a_shaping_w1.py --validate-only
uv run python scripts/test_phase3a_shaping_w1.py --device cpu --n-seeds 5
```

## Remote GPU

```bash
ssh -A anaderi@89.169.123.173  # NVIDIA L40S
cd projects/graph-fans && git pull && source ~/.local/bin/env
```

## Key references

- `LOG.md` — Full experiment log with results tables
- `CONTEXT.md` — Detailed context for new agents (root causes, fix plan, design decisions)
- `plans/Roadmap-Hypotheses-Validation.md` — Full H1-H6 research roadmap
- Vault: `~/docker-stacks/obsidian/config/Desktop/vault/1-Project/2026-GraphFANS/`
