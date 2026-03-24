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

## 2026-03-23 — Phase 2c: Alt-4 (Spectral Fidelity Loss) — G2: NO-GO

Added a differentiable spectral fidelity loss via Tweedie denoising: at each training step, compute the one-step denoised estimate x̂₀ = (x_t + σ²·score)/μ, then penalize per-band energy mismatch |E_b(x̂₀) - E_b(x₀)| weighted by importance weights. Combined with Alt-2 (cosine schedule + EMA + LR annealing), 1000 epochs, spectral_loss_weight=0.1.

### Results

| Family | Uniform QBE | Spectral+Loss QBE | Change |
|--------|------------|-------------------|--------|
| SBM(q=0.05) | 0.062 | 0.065 | -4.2% |
| BA(m=5) | 0.125 | 0.118 | +5.3% |
| Cora | 0.062 | 0.064 | -2.9% |

H2: ρ=−0.21, p=0.74 (identical to 2b — spectral loss only affects the "spectral" H1-A runs).

**G2 gate: NO-GO** — Results are statistically indistinguishable from Phase 2b (no spectral loss).

### Key insight
The spectral loss term had **zero additional effect** beyond what Alt-2 already provided. The Tweedie denoised estimate x̂₀ at intermediate timesteps is too noisy for the per-band energy comparison to provide a useful gradient signal. The loss term effectively adds noise to the gradient rather than useful spectral supervision.

### Overall Phase 2 Conclusion

Three experiments (2a, 2b, 2c) tested spectral noise shaping with increasing sophistication:
- **2a (baseline):** Unstable training, SBM degraded. NO-GO.
- **2b (cosine+EMA):** Stable training, shaping is a no-op. NO-GO.
- **2c (cosine+EMA+spectral loss):** No additional benefit from explicit supervision. NO-GO.

**Root cause:** The 3-layer GCN score network on 200-node graphs operates at a scale where spectral noise shaping provides no signal. The model's bottleneck is not spectral fidelity of the noise — it's the expressiveness and training dynamics of the score estimator itself. At this scale, uniform noise is sufficient because the model cannot resolve spectral structure regardless.

**Remaining options:**
1. **Scale up:** Test on larger graphs (>1000 nodes) with deeper networks (6+ layers, Graph Transformer). The FANS paper showed benefits only at sufficient model capacity.
2. **Alternative 5 (Direct spectral generation):** Generate per-band coefficients independently, bypassing the score network's inability to differentiate bands. This is a fundamentally different architecture.
3. **Write up negative result:** Document that the FANS→graph transfer does not work at small scale with simple score networks. This is a publishable negative result.

Results: `results/phase2c/`

## 2026-03-23 — Phase 2d: Multiscale Features — G2: NO-GO

Switched from smooth features to multiscale features (community-mode) with role-dependent generation. Cosine SDE + EMA + LR annealing. 1000 epochs, 3 seeds, NVIDIA L40S GPU.

Results identical to 2b/2c — spectral noise shaping remains a no-op.

Results: `results/phase2d/`

## 2026-03-24 — Phase 2e: Dataset Training + Community Features — G2: NO-GO

### Config
Cosine SDE + EMA + LR annealing + 500-sample feature dataset per graph + community features. 1000 epochs, 3 seeds, NVIDIA L40S GPU.

### Results

| Family | Uniform QBE (high bands) | Spectral QBE | p-value |
|--------|-------------------------|-------------|---------|
| SBM(q=0.05) | 0.056 ± 0.000 | 0.056 ± 0.000 | 0.79 |
| BA(m=5) | 0.009 ± 0.008 | 0.012 ± 0.011 | 0.76 |

H2: ρ=−0.29, p=0.64. G2: NO-GO (0/2 families pass).

Dataset fix improved absolute generation quality 14× (BA QBE 0.009 vs 0.125 in 2b), but uniform and spectral remain indistinguishable.

### Critical Bug Discovered: Model Generates Pure Noise

Post-hoc investigation revealed generated features have spectral profile matching random noise, not training data:

    ref profile:   [0.386  0.  0.376  0.01  0.058  0.064  0.065  0.041]  ← bimodal (community)
    gen profile:   [0.054  0.  0.016  0.035  0.217  0.258  0.246  0.173]  ← matches random noise
    gen std: 13.7 vs ref std: 1.3 (10× too high)

The "identical QBE" across all experiments was measuring noise-vs-noise distance. ALL Phase 2 results (2a–2e) are invalid.

### Root Causes Confirmed

1. **Wrong training target** (`trainer.py:187`): `target = -noise / std` (score formulation) — at small std near t=0, target explodes. Should be ε-prediction: `target = noise`
2. **Reverse SDE adds noise at t→0** (`sde.py` reverse_step): stochastic noise at every step including final, corrupting output. Should use DDIM (deterministic) with Tweedie at t=0
3. **Insufficient training**: 8,000–16,000 gradient steps = 16 passes/sample. Need ~64,000 (128 passes)
4. **No dataset persistence**: features generated on-the-fly, impossible to inspect before training

### Fix Plan

See `~/.claude/plans/zany-churning-dolphin.md` and `~/projects/graph-fans/CONTEXT.md`.

Results: `results/phase2e/`

## 2026-03-24 — Phase 2f Prep: Code Fixes + Dataset Pre-generation

### Code Fixes Applied

Fixed 4 root causes from Phase 2e investigation:

1. **ε-prediction** (`trainer.py`): Training target changed from `-noise/std` (score) to `noise` (epsilon). Eliminates target explosion at small std near t=0.
2. **DDIM sampling** (`sde.py`, `trainer.py`): Added `alpha_bar()` + `ddim_step()` to both SDE classes. Generation now uses deterministic DDIM reverse steps with Tweedie formula at t=0 — no stochastic noise corruption.
3. **Training budget** (`trainer.py`): Defaults increased to 2000 epochs × 32 timesteps = 64,000 steps (128 passes/sample, was 16).
4. **Dataset persistence** (`dataset.py`, new): Pre-generate + cache train/ref datasets as `.npz`. Enables inspection before training.

Also added: post-training sanity check (compares gen std and spectral profile to training data), Tweedie formula fix in `spectral_loss.py`.

Tests: 59/59 passing (31 Phase 2 tests, up from 18).

### Dataset Inspection (5 families × 3 seeds)

All datasets: 500 train + 50 ref samples, 200 nodes, 16 features, community mode.

| Family | Train Std | Spectral Profile (8 bands) | Max Ratio | Bimodal |
|--------|-----------|---------------------------|-----------|---------|
| SBM(q=0.01) | 1.21 | [0.805  0  0  0.019  0.042  0.052  0.049  0.034] | 43× | Yes |
| SBM(q=0.05) | 1.28 | [0.387  0  0.373  0.010  0.059  0.064  0.065  0.042] | 37× | Yes |
| SBM(q=0.1) | 1.34 | [0.437  0  0  0.274  0.064  0.088  0.082  0.055] | 8× | Yes |
| BA(m=2) | 1.08 | [0.318  0.450  0.088  0.037  0.047  0.023  0.021  0.016] | 28× | No |
| BA(m=5) | 1.17 | [0.264  0  0.407  0.178  0.065  0.038  0.031  0.017] | 24× | Yes |
| **Random noise** | — | [0.007  0  0.016  0.037  0.221  0.274  0.251  0.195] | — | — |

**Key observations:**
- Features have std ≈ 1.1–1.3 (healthy), mean ≈ 0 (centered)
- Spectral energy concentrated in **low/mid bands** (community structure), opposite to random noise (high bands)
- Profiles stable across seeds (same graph topology, different feature realizations)
- SBM(q=0.01) most extreme: 80% energy in band 0 (nearly disconnected communities)
- BA(m=2) not bimodal but still highly non-uniform (77% in bands 0–1)

All graphs connected, density 2–15%, spectral gaps 0.06–0.46.

Datasets: `results/phase2f/datasets/`
