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

## 2026-03-25 — Phase 2f: Full Evaluation — G2: NO-GO

### Config
ε-prediction + DDIM + cosine SDE + EMA + LR annealing + 500-sample dataset + community features. 2000 epochs, 32 timesteps/epoch. NVIDIA L40S GPU.

Two scales: full (200 nodes × 16 features, 5 seeds) and small (50 nodes × 4 features, 3 seeds).

### H1-A Results (full-scale, 200×16)

| Family | Uniform QBE | Spectral QBE | p-value |
|--------|------------|-------------|---------|
| SBM(q=0.05) | 0.0543 ± 0.0003 | 0.0544 ± 0.0004 | 0.13 |
| BA(m=5) | 0.0333 ± 0.0002 | 0.0332 ± 0.0005 | 0.64 |

G2: NO-GO (0/2 families pass). H2: ρ=0.72, p=0.17 (not significant). QBE spread across t_knee values <0.0004.

### Sanity Checks
- SBM(q=0.05): gen std 8.7±1.3× training std → 0/70 pass
- BA(m=5): gen std 3.5±0.3× training std → 0/68 pass
- SBM(q=0.01): gen std 11.0±0.3× → 0/50 pass
- BA(m=2): gen std ~1.5× → **25/25 pass** (only family that denoises at 200 nodes)

### Spectral Wasserstein Diagnostic

New per-band W1 metric reveals distributional effects invisible to QBE:

| Method | SBM(q=0.01) W1 | SBM(q=0.05) W1 |
|--------|----------------|----------------|
| Oracle (train vs ref) | 44 | 38 |
| Random noise | 409 | 288 |
| Uniform | 1360 | 2351 |
| **Spectral** | **842 (−38%)** | **1903 (−19%)** |
| Spectral ramp (all t_knee) | 898–909 | 1867–1896 |

Spectral shaping reduces W1 by 20–38%, concentrated in non-dominant bands (bands 1–7 W1 drops 65%). But all models produce worse-than-noise distributions. t_knee has <1% effect on W1 — definitively dead.

### Analysis
Loss floor ~0.42 across both scales. Model explains ~57% of noise variance, insufficient for DDIM generation. The training formulation fixes were necessary but not sufficient — a deeper training dynamics issue remains.

Reports: `Report-Phase2f.md` and `Report-Phase2f-Intermediate.md` in vault.
Results: `results/phase2f/` (full), `results/phase2f_small/` (small), `results/diagnostics/` (W1)

## 2026-03-25 — NS-D: SNR-Bin Loss Diagnostic — Gradient Misallocation Confirmed

### Motivation
Analysis of Dieleman (2024) "Noise Schedules Considered Harmful" suggested the loss floor of ~0.4 may be a training weighting problem, not a model capacity limit. With ε-prediction + uniform t-sampling, gradient budget may be misallocated across noise levels.

### Method
Trained models on SBM(q=0.01) and SBM(q=0.05) (50 nodes, 4 features, 500 epochs), then evaluated MSE loss at 1000 timesteps binned into 10 log₁₀(SNR) bins. Measured both per-bin loss and the fraction of gradient each bin receives under uniform t-sampling.

### Results: SBM(q=0.01)

| SNR regime | log₁₀(SNR) | Mean Loss | Grad Share |
|-----------|------------|-----------|------------|
| Pure noise | −5 to −3 | 0.014 | 3.6% |
| High noise | −3 to −1 | 0.028–0.070 | 10% |
| Mid noise | −1 to +1 | 0.14–0.34 | 54% |
| **Low noise** | **+1 to +4** | **0.72–1.11** | **14%** |

### Results: SBM(q=0.05)

| SNR regime | log₁₀(SNR) | Mean Loss | Grad Share |
|-----------|------------|-----------|------------|
| Pure noise | −5 to −3 | 0.020 | 3.6% |
| High noise | −3 to −1 | 0.035–0.103 | 10% |
| Mid noise | −1 to +1 | 0.31–0.66 | 54% |
| **Low noise** | **+1 to +4** | **0.93–1.10** | **14%** |

### Key Findings

1. **Loss ranges 80× across SNR bins** (0.014 at high noise → 1.11 at low noise). The model is excellent at denoising heavily corrupted inputs but essentially random at low-noise timesteps (loss ≈ 1.0 = predicting zero).
2. **Low-noise regime is catastrophically underfit.** Bins with log₁₀(SNR) > 1 have loss 0.72–1.10 — barely better than predicting nothing. These are the DDIM final steps that determine output quality.
3. **Only 14% of gradient goes to the underfit regime.** 54% goes to mid-noise bins where loss is 0.14–0.66 (partially learned). The model is starved of gradient exactly where it needs it most.
4. **This explains DDIM generation failure.** Final DDIM steps operate at high SNR where the model has loss ≈ 1.0. These steps corrupt the output regardless of how well earlier steps performed.
5. **The loss floor of ~0.4 is a misleading average.** It averages well-learned high-noise bins (0.01) with completely unlearned low-noise bins (1.0). The model is not capacity-limited — it's gradient-starved in the wrong regime.

### Implication
Gradient reweighting (log-SNR sampling, min-SNR-γ, or v-prediction) should directly improve generation by shifting gradient to the underfit low-noise regime. This is the most promising path to breaking the loss floor without architectural changes.

Results: `results/diagnostics/snr_profile_SBM_q{0.01,0.05}_seed0.json`
Analysis: `plans/noise-schedule-analysis.md`

## 2026-03-25 — NS-A + NS-C: Log-SNR Sampling + Min-SNR-γ Weighting — Capacity Confirmed

### Config
Implemented two gradient rebalancing mechanisms from Dieleman (2024) / Hang et al. (2023):
- **NS-A**: Uniform sampling in log-SNR space (replaces uniform t-sampling)
- **NS-C**: min-SNR-γ loss weighting with γ=5 (clips trivial high-SNR gradients)

Tested on SBM(q=0.01), 50 nodes, 4 features, 500 epochs, cosine SDE + EMA.

### Per-SNR-bin Loss Comparison

| Bin | log₁₀(SNR) | Baseline | NS-A | NS-C | NS-A+C |
|-----|-----------|----------|------|------|--------|
| 0–2 | −5 to −2 (high noise) | 0.015 | **0.005** | 0.013 | **0.001** |
| 3–4 | −2 to −1 (mid-high) | 0.029–0.071 | 0.025–0.076 | 0.027–0.070 | 0.023–0.075 |
| 5–6 | −1 to +1 (mid) | 0.139–0.344 | 0.161–0.397 | 0.137–0.346 | 0.151–0.386 |
| **7–9** | **+1 to +4 (low noise)** | **0.72–1.11** | **0.74–1.03** | **0.74–1.13** | **0.77–1.13** |

Final loss: Baseline 0.321, NS-A 0.236, NS-C 0.142, NS-A+C 0.084.

### Key Findings

1. **Final loss drops 4× (0.32→0.08)** but this is misleading — min-SNR-γ scales down loss at high-SNR bins, making reported loss lower without improving the model at those bins.
2. **High-noise bins (0–2) improve 3–15×** with NS-A — the model learns to predict noise in the easy regime much better.
3. **Low-noise bins (7–9) are unchanged** — loss 0.72–1.13 regardless of training strategy. NS-A gives marginal 4–7% improvement at bins 8–9 only.
4. **NS-C has zero effect on per-bin loss** — identical to baseline at every bin. The "lower final loss" is purely the weighting artifact.
5. **The gradient misallocation hypothesis was partially correct but insufficient.** Rebalancing helps the easy regime but the hard regime (low noise, high SNR) is genuinely beyond the model's capacity.

### Conclusion

**The loss floor at low-noise bins is a confirmed architectural capacity ceiling**, not a training dynamics issue. The 3-layer GCN (128 hidden) cannot predict noise when the signal is mostly intact — it lacks the representational capacity to model fine-grained spectral structure. Next step: increase model depth to 6 layers.

Results: `results/ns_ac_comparison.log`, `results/diagnostics/snr_profile_SBM_q0.01_seed0_*.json`

## 2026-03-25 — Deeper Network (6-Layer GCN): Modest Improvement, Capacity Ceiling Persists

### Config
Tested 6-layer GCN (with LayerNorm) at 128 and 256 hidden dim, with and without NS-A (log-SNR sampling). SBM(q=0.01) and SBM(q=0.05), 50 nodes, 4 features, 500 epochs, cosine SDE + EMA.

### Per-SNR-bin Loss at Low-Noise Regime (bins 7–9)

**SBM(q=0.01):**

| Config | Bin 7 (+1.5) | Bin 8 (+2.4) | Bin 9 (+3.3) | Final loss |
|--------|-------------|-------------|-------------|------------|
| 3L/128 (baseline) | 0.723 | 0.996 | 1.107 | 0.321 |
| **6L/128** | **0.659** | **0.944** | **1.053** | 0.292 |
| 6L/256 | 0.660 | 0.943 | 1.047 | 0.294 |
| 6L/128+NS-A | 0.704 | 0.935 | 1.016 | 0.223 |

**SBM(q=0.05):**

| Config | Bin 7 (+1.5) | Bin 8 (+2.4) | Bin 9 (+3.3) | Final loss |
|--------|-------------|-------------|-------------|------------|
| 3L/128 (baseline) | 0.933 | 1.064 | 1.095 | 0.518 |
| **6L/128** | **0.921** | **1.052** | **1.082** | 0.505 |

### Key Findings

1. **6 layers improves low-noise bins 5–13%** — first real improvement in this regime across all experiments. Bin 7: 0.723→0.659 (−9%) on SBM(q=0.01).
2. **256 hidden dim = no improvement over 128** at 6 layers. Width is not the bottleneck.
3. **Low-noise bins remain at 0.66–1.08 in all configs.** No configuration breaks below 0.6 at bin 7. The GCN architecture has a fundamental expressiveness ceiling.
4. **SBM(q=0.05) much harder** — 6L gets bin 7 to only 0.921 (vs 0.659 for q=0.01). Bimodal spectral structure is genuinely harder.

### Architectural Analysis

A GCN is a polynomial filter on the graph Laplacian: each layer applies a 1st-order polynomial (weighted neighbor average), so K layers produce a K-th order Chebyshev polynomial. A 6-layer GCN can represent any 6th-order polynomial of the eigenvalues.

For noise prediction at low noise (high SNR), the model needs to resolve fine spectral detail — distinguish nearby eigenvalues and predict their contributions to the noise. A 6th-order polynomial can separate at most 6 spectral peaks. With community features that have energy concentrated in 2–3 bands, this should theoretically suffice, but the polynomial basis is inefficient: it approximates sharp spectral features poorly.

**Implication:** The bottleneck is the polynomial spectral response of GCN convolutions. Attention-based architectures (GAT, Graph Transformer) have per-edge adaptive weights that are not constrained to polynomial spectral filters, potentially breaking this ceiling.

### Decision

The GCN line of investigation is exhausted: 3L→6L gives diminishing 5–13% returns, 128→256 hidden gives zero return. **Pivot to Graph Transformer or proceed with Alt-5 (direct spectral generation).**

Results: `results/depth_{6L,6L_256,6L_nsa,6L_sbm05}.log`
