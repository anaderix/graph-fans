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

## 2026-03-23 — Phase 2: Regime A Core — G2: NO-GO

### Implementation
- 8 modules: noise_shaper, score_network (3-layer GCN), sde (VP-SDE), trainer, evaluate, visualize, run_experiment, tests
- Code reviewed by tester agent: 1 CRITICAL fixed (G2 gate ignoring Bonferroni), 2 MAJOR fixed (H2 direction check, misleading same-graph metrics)
- 18/18 tests passing
- Ran on NVIDIA L40S GPU (~90 min)

### Phase 2: H1-A (Spectral Band Decomposition) — FAIL

| Family | Uniform QBE | Spectral QBE | Change |
|--------|------------|-------------|--------|
| SBM(q=0.05) | 0.049 | 0.109 | -120% (worse) |
| BA(m=5) | 0.136 | 0.136 | +0.2% |
| Cora | 0.087 | 0.088 | -0.4% |

0/3 families pass Bonferroni-corrected significance test. Spectral noise shaping actively hurt SBM and had no effect on BA/Cora.

### Phase 2: H2 (Temporal Ramp) — FAIL

Spearman ρ=0.22, p=0.72, 95% CI=[−1.0, 1.0]. No correlation between optimal t_knee and spectral gap ratio. QBE curves are flat across all t_knee values.

**G2 gate: NO-GO** — Spectral noise shaping does not improve feature generation with this simplified score model.

### Diagnosis
- 3-layer GCN likely lacks capacity to exploit spectral noise structure
- VP-SDE training is unstable (loss spikes) even with gradient clipping
- Temporal ramp has no effect because the model can't use spectral structure at all

### Recommended alternatives (ranked by effort)
1. **Noise schedule recalibration** — sub-VP SDE, cosine schedule, EMA (1 day)
2. **Spectral loss term** — directly penalize per-band energy mismatch (2-3 days)
3. **Larger score network** — Graph Transformer / deeper GCN (1-2 days)
4. **Per-instance importance weights** — adaptive instead of family-averaged (2-3 days)
5. **Direct spectral generation** — generate in eigenbasis, per-band diffusion (1-2 weeks)

Results: `results/phase2/`

## 2026-03-23 — Phase 2b: Alt-2 (Cosine Schedule + EMA + LR Annealing) — G2: NO-GO

Implemented and tested Alternative 2: cosine noise schedule (Nichol & Dhariwal), EMA (decay=0.999), cosine LR annealing, 1000 epochs. Run on NVIDIA L40S.

### Results

| Family | Uniform QBE | Spectral QBE | Change |
|--------|------------|-------------|--------|
| SBM(q=0.05) | 0.064 | 0.065 | -2.3% |
| BA(m=5) | 0.125 | 0.118 | +5.3% |
| Cora | 0.071 | 0.072 | -1.8% |

H2: Spearman ρ=−0.21, p=0.74. No correlation.

**G2 gate: NO-GO** (0/3 families pass Bonferroni-corrected significance test)

### Key insight
Alt-2 **fixed the training stability** (SBM spectral QBE improved from 0.109→0.065, no longer degraded), but spectral noise shaping is now a **no-op** — uniform and spectral are statistically indistinguishable. The model trains stably with shaped noise but doesn't benefit from it.

### Conclusion
The problem is not training instability. The 3-layer GCN score network simply cannot exploit spectral noise structure. Next step: try Alternative 4 (spectral loss term) which directly supervises spectral fidelity rather than relying on the model to learn it implicitly from shaped noise.

Results: `results/phase2b/`
