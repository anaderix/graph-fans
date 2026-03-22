# Graph-FANS Project Log

## 2026-03-22 — Project Initialization & Phase 0 + Phase 1a

### Project setup
- Initialized Python project at `~/projects/graph-fans` with `uv`
- Dependencies: numpy, scipy, networkx, torch, torch-geometric, matplotlib, seaborn, scikit-learn, pandas
- Package structure: `graph_fans/{utils,phase0,phase1a}`, `tests/`, `results/`

### Phase 0: Empirical Spectral Profiling — G0: GO

Profiled spectral energy distribution across 9 graph families (3 seeds each):

| Family | Energy Ratio | Decay Model | R² |
|--------|-------------|-------------|-----|
| SBM(q=0.01) | 3.8× | power_law | 0.12 |
| SBM(q=0.05) | 6.9× | power_law | 0.35 |
| SBM(q=0.1) | 10.7× | power_law | 0.44 |
| SBM(q=0.2) | 8.3× | power_law | 0.51 |
| BA(m=2) | 17.0× | exponential | 0.39 |
| BA(m=5) | 8.1× | power_law | 0.05 |
| BA(m=10) | 6.7× | power_law | 0.30 |
| Cora | 4.0× | power_law | 0.49 |
| CiteSeer | 5.0× | power_law | 0.39 |

**G0 gate: GO** — All 9/9 families exceed the 2× energy ratio threshold.

**Key finding:** Graph spectral energy is non-uniform but does NOT follow a simple monotonic decay like images (1/f). Energy profiles are hump-shaped or concentrated in mid-bands for several families. This means data-driven per-family weighting is more appropriate than a blanket inverse-power scheme.

Results: `results/phase0/`

### Phase 1a: Basis-Free Spectral Metrics — G1: GO

Implemented and validated 3 basis-free spectral metrics:
1. **Spectral density JSD** — KDE on normalized eigenvalues
2. **Quantile-band energy (QBE) distance** — per-band energy comparison
3. **Heat kernel signature (HKS) distance** — Tr(exp(-tL)) curves

Compared against standard MMD metrics (degree, clustering, triangle) on 12 reference-baseline pairs.

**G1 gate: GO** — QBE distance consistently reveals spectral fidelity gaps invisible to standard MMD, especially for configuration-model baselines (matched degree sequence but wrong spectral structure).

Results: `results/phase1a/`

### Tests
- 28/28 tests passing (`tests/test_phase0.py`, `tests/test_phase1a.py`)
