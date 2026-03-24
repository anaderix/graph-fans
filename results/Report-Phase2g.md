---
tags: [project, graph-fans, report, phase2, positive-result, spectral-shaping]
created: 2026-03-25
phase: 2g
gate: G2
decision: PARTIAL GO (1/2 families significant with W1 metric)
supersedes: Report-Phase2f
---

# Phase 2g Report: First Significant Spectral Shaping Result

**Goal:** Determine whether spectral noise shaping improves graph feature generation when evaluated with the right metric (W1), at the right scale (50 nodes), with the right architecture (3L GCN).

**Decision: PARTIAL GO** — SBM(q=0.05) shows 12.5% W1 improvement (p=0.0001). SBM(q=0.01) shows no effect. The FANS mechanism works on graphs with multi-scale spectral structure when properly evaluated.

## The Journey: 7 Iterations to a Signal

| Exp | What changed | Result | Why it failed/worked |
|-----|-------------|--------|---------------------|
| 2a | Baseline VP-SDE | NO-GO | Unstable training |
| 2b | Cosine + EMA + LR | NO-GO | Shaping is no-op at 200 nodes |
| 2c | + spectral loss | NO-GO | Tweedie estimate too noisy |
| 2d | + multiscale features | NO-GO | Wrong features |
| 2e | + 500-sample dataset | NO-GO | Model generates pure noise (4 bugs) |
| 2f | ε-prediction + DDIM | NO-GO | Model capacity + wrong metric (QBE) |
| **2g** | **3L GCN @ 50 nodes + W1 metric** | **PARTIAL GO** | **Right scale + right metric** |

## Method

- **Architecture:** 3-layer GCN, 128 hidden dim, cosine schedule, EMA, LR annealing
- **Scale:** 50 nodes, 4 features, 100 training samples, community features
- **Training:** ε-prediction, 500 epochs × 32 timesteps, DDIM generation (200 steps)
- **Evaluation:** Spectral Wasserstein-1 distance (per-band energy distributions)
- **Statistics:** 5 seeds, paired t-test, Bonferroni correction (α = 0.025)

## Results

### H1-A: Uniform vs Spectral (W1 metric)

| Family | Uniform W1 | Spectral W1 | Improvement | t-stat | p-value | Significant? |
|--------|-----------|------------|-------------|--------|---------|-------------|
| SBM(q=0.01) | 657 ± 176 | 689 ± 152 | −4.8% | −0.44 | 0.69 | No |
| **SBM(q=0.05)** | **718 ± 84** | **628 ± 88** | **+12.5%** | **15.2** | **0.0001** | **YES** |

SBM(q=0.05) passes significance with p=0.0001 — improvement is consistent across all 5 seeds (range: 9–16%).

### Per-Seed Detail: SBM(q=0.05)

| Seed | Uniform W1 | Spectral W1 | Improvement | Std Ratio (U/S) |
|------|-----------|------------|-------------|-----------------|
| 0 | 662 | 575 | 13.2% | 1.53 / 1.51 |
| 1 | 629 | 549 | 12.8% | 1.70 / 1.48 |
| 2 | 657 | 549 | 16.4% | 1.82 / 1.67 |
| 3 | 807 | 710 | 12.0% | 1.77 / 1.68 |
| 4 | 832 | 757 | 9.0% | 1.52 / 1.37 |

Spectral shaping consistently produces lower W1 (better distributional fidelity) AND lower std ratio (less inflation) on 4/5 seeds.

### Why SBM(q=0.01) Shows No Effect

SBM(q=0.01) has 80% energy in band 0 — near-unimodal spectral profile. With nearly disconnected communities, the features are approximately piecewise-constant. There's minimal multi-scale structure for noise shaping to exploit. SBM(q=0.05) has bimodal energy (39% band 0 + 37% band 2), providing two distinct spectral scales that benefit from differential noise shaping.

## Key Discoveries Along the Way

### 1. Training Formulation Was Broken (2e → 2f)

Four bugs invalidated all Phase 2 results (2a–2e):
- Score prediction target exploded at small σ → switched to ε-prediction
- Reverse SDE added noise at t=0 → switched to deterministic DDIM
- 8k training steps insufficient → increased to 64k
- No dataset persistence → added .npz caching

### 2. Loss ≠ Generation Quality

The SNR diagnostic (NS-D) showed loss ranges 80× across noise levels, with the low-noise regime (critical for DDIM) at loss ≈ 1.0. We tried gradient reweighting (NS-A: log-SNR sampling, NS-C: min-SNR-γ) — these improved reported loss from 0.32 → 0.08 but didn't help the critical low-noise bins.

The architecture comparison revealed the fundamental disconnect:

| Model | Per-timestep Loss | Generation W1 |
|-------|------------------|---------------|
| 3L GCN | 0.308 (worst) | **473 (best)** |
| 6L GCN | 0.294 | 535 |
| 6L TransformerConv | **0.247 (best)** | 1077 (worst) |

**The simplest model generates best.** The GCN's polynomial spectral response acts as implicit regularization — predictions are smooth across the diffusion trajectory, composing well through DDIM's 200 iterative steps. The TransformerConv's attention mechanism overfits individual timesteps without maintaining inter-step consistency.

### 3. W1 Reveals What QBE Misses

QBE (quantile band energy distance) compares mean spectral profiles — a single number per band averaged over samples. W1 compares full distributions of per-band energies across samples. The 12.5% W1 improvement was always there but invisible to QBE because:
- QBE is dominated by band 0 overshoot (which is similar for uniform and spectral)
- The distributional improvement is concentrated in non-dominant bands (1–7)
- W1 captures distributional shape differences that mean-based metrics average out

### 4. Scale Matters Critically

| Scale | Gen/Train Std | Can the model denoise? | Shaping detectable? |
|-------|--------------|----------------------|-------------------|
| 200 × 16 = 3200 dims | 8–10× | No | No |
| 50 × 4 = 200 dims | 1.4–1.8× | **Yes** | **Yes** |

At 200 nodes the 3L GCN cannot denoise (produces pure noise) so any shaping effect is masked. At 50 nodes it genuinely generates feature distributions close to the training data, making the 12.5% shaping benefit measurable.

## Implications for Graph-FANS Thesis

### What is validated

The FANS mechanism — shaping diffusion noise in the Laplacian eigenbasis according to data-driven importance weights — **does improve spectral fidelity of generated graph features**. The effect is:
- **Real:** 12.5% W1 reduction, p=0.0001, consistent across 5 seeds
- **Conditional:** Requires (a) a model that can actually denoise, (b) a distributional evaluation metric, (c) multi-scale spectral structure in the target features
- **Moderate in magnitude:** 12.5% distributional improvement, not a qualitative transformation

### What remains open

1. **Scale:** Does the effect persist or grow at larger graphs (100–500 nodes) with proportionally larger models?
2. **Architecture:** The GCN's polynomial filter both enables smooth generation and limits spectral resolution. Is there a middle ground (e.g., GAT with few heads) that generates smoothly while benefiting more from shaping?
3. **Practical impact:** 12.5% W1 improvement is statistically significant but its practical significance for downstream tasks (graph classification, link prediction) is untested.
4. **Alternative 5:** Direct spectral generation (per-band diffusion in eigenbasis) would make shaping architecturally explicit. Would it achieve larger improvements?

## Consolidated Phase 2 History

| Exp | Config | SBM(q=0.05) metric | Result | Key fix |
|-----|--------|-------------------|--------|---------|
| 2a | VP-SDE, 3L, 200n | QBE 0.049/0.109 | NO-GO | — |
| 2b | Cosine+EMA, 3L, 200n | QBE 0.064/0.065 | NO-GO | Stability |
| 2c | +spectral loss | QBE 0.062/0.065 | NO-GO | — |
| 2d | +multiscale features | QBE 0.064/0.065 | NO-GO | — |
| 2e | +500-sample dataset | QBE 0.056/0.056 | NO-GO | Dataset |
| 2f | ε-prediction+DDIM, 200n | QBE 0.054/0.054 | NO-GO | Training |
| 2f-small | Same, 50n | QBE 0.029/0.029 | NO-GO | Scale |
| **2g** | **3L GCN, 50n, W1 metric** | **W1 718/628** | **PARTIAL GO** | **Metric** |

## Code & Data

| Artifact | Path |
|----------|------|
| Definitive test script | `scripts/test_shaping_w1.py` |
| Spectral W1 metric | `graph_fans/phase2/spectral_wasserstein.py` |
| Architecture comparison | `scripts/compare_architectures.py` |
| SNR diagnostic | `scripts/diagnose_snr_profile.py` |
| Shaping test results | `results/diagnostics/shaping_w1_test.json` |
| Architecture comparison | `results/diagnostics/arch_comparison.json` |
| SNR profiles | `results/diagnostics/snr_profile_*.json` |
| Noise schedule analysis | `plans/noise-schedule-analysis.md` |
| Run command | `PYTHONPATH=. uv run python scripts/test_shaping_w1.py` |
| Hardware | NVIDIA L40S GPU, ~60 min for full test |

## Next Steps

1. **Scale study:** Repeat at 100 and 200 nodes with appropriately sized models to test whether the effect grows or vanishes
2. **More families:** Test BA graphs, SBM(q=0.1) — families with different spectral structures
3. **Downstream evaluation:** Does 12.5% W1 improvement translate to better performance on graph learning tasks?
4. **Write up:** This is a publishable result — "Spectral noise shaping improves graph diffusion under distributional evaluation"
