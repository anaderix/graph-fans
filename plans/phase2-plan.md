---
roadmap: 1-Project/2026-GraphFANS/Roadmap-Hypotheses-Validation.md
scope: phase2
created: 2026-03-22
status: approved
---

# Execution Plan: Phase 2 — Regime A Core (H1-A + H2)

## Overview

Phase 2 implements and validates Graph-FANS noise shaping on fixed-topology graphs. A simplified score-based diffusion model generates node features (not topology) on fixed graphs. We compare uniform Gaussian noise against spectrally-shaped noise (H1-A) and test whether the optimal transition timestep t_knee correlates with the spectral gap ratio (H2). This is the core validation of the Graph-FANS thesis.

## Prerequisites (Confirmed)

- **G0: GO** — 9/9 families show non-uniform spectral energy (3.8×–17.0×)
- **G1: GO** — QBE distance reveals spectral gaps invisible to standard MMD (up to 63×)
- **Phase 0 output:** `results/phase0/band_energies.json` available for importance weights
- **Phase 1a output:** `SpectralMetrics` class ready for evaluation

## Phase 2: Regime A Core

### Objective

Prove that shaping diffusion noise according to the spectral energy profile improves node feature generation fidelity (H1-A), and that the optimal transition timestep correlates with graph spectral properties (H2).

### Implementation Tasks

#### Task 1: `graph_fans/phase2/noise_shaper.py` — Core FANS Mechanism

- `compute_importance_weights(band_energies, alpha=1.0, epsilon=1e-3) -> ImportanceWeights`
  - Inverse-power weights: g_b = (pi_bar_b + epsilon)^(-alpha)
  - Uses empirical energies from Phase 0, NOT assumed functional forms
- `shape_noise(noise, eigenvectors, band_indices, weights) -> Tensor`
  - Project into eigenbasis, scale per band, project back
  - **Critical:** per-band unit-variance normalization after scaling
- `shape_noise_with_temporal_ramp(noise, eigenvectors, band_indices, weights, t, t_knee) -> Tensor`
  - phi(t) smooth interpolation: uniform below t_knee, shaped above
  - Smooth transition via cosine schedule to avoid loss discontinuities

#### Task 2: `graph_fans/phase2/score_network.py` — Score Network

- `SimpleScoreNetwork(n_features, hidden_dim=128, n_layers=3, time_emb_dim=32)`
  - 3-layer GCN with sinusoidal timestep embedding
  - Skip connections between layers
  - Input: noisy features + t → score estimate (same shape)
  - Uses PyTorch Geometric `GCNConv`

#### Task 3: `graph_fans/phase2/sde.py` — SDE Framework

- `VPSDE(beta_min=0.1, beta_max=20.0, T=1.0, n_timesteps=1000)`
  - Linear beta schedule, VP-SDE formulation
  - `marginal_params(t) -> (mean_coeff, std)` for q(x_t | x_0)
  - `perturb(x_0, t, noise=None) -> (x_t, noise_used)`
  - `reverse_step(x_t, score, t, dt) -> x_t-dt`

#### Task 4: `graph_fans/phase2/trainer.py` — Training Loop

- `TrainConfig(n_epochs=500, lr=1e-3, batch_timesteps=16, seed=42, use_spectral_noise=False, t_knee=None, alpha=1.0, B=8)`
- `Trainer(config, graph, features, importance_weights=None)`
  - Denoising score matching: L = E_t E_{x0} E_{noise} [ ||score(x_t, t) - noise/std||^2 ]
  - When `use_spectral_noise=True`, noise is shaped via noise_shaper
  - `train() -> dict` with loss history
  - `generate(n_steps=200) -> ndarray` via reverse SDE

#### Task 5: `graph_fans/phase2/evaluate.py` — Evaluation Pipeline

- `Phase2Results` dataclass with per-band QBE, aggregate metrics, training info
- `evaluate_single(graph, features_ref, features_gen, ...) -> Phase2Results`
- `run_h1a_experiment(graph_families, n_seeds=5, ...) -> DataFrame`
  - 3 families × 2 methods × 5 seeds = 30 runs
- `run_h2_experiment(graph_families, t_knee_values, n_seeds=5, ...) -> DataFrame`
  - 5 families × 5 t_knee × 5 seeds = 125 runs

#### Task 6: `graph_fans/phase2/visualize.py` — Plots

- `plot_per_band_comparison` — grouped bar: uniform vs spectral per band
- `plot_h1a_summary` — heatmap of improvement per family
- `plot_t_knee_grid` — QBE vs t_knee lines per family
- `plot_h2_correlation` — scatter: optimal t_knee vs λ₂/λ_max with Spearman ρ
- `plot_g2_summary` — gate decision figure

#### Task 7: `graph_fans/phase2/run_experiment.py` — Main Script

- Full pipeline: load Phase 0 weights → H1-A experiment → H2 experiment → G2 decision → plots
- argparse with `--output-dir`, `--n-nodes`, `--seeds`, `--epochs`, `--t-knee-values`

#### Task 8: `tests/test_phase2.py` — Tests

### Data Requirements

| Family | Parameters | Role |
|--------|-----------|------|
| SBM(q=0.05) | n=200, 4 communities, p_intra=0.3 | H1-A primary |
| BA(m=5) | n=200 | H1-A primary |
| Cora | n=2708, real features | H1-A primary |
| SBM(q=0.01,0.1,0.2) | n=200 | H2 grid |
| BA(m=2,10) | n=200 | H2 grid |

**Feature protocol:** Training features seed=S, reference features seed=S+1000 (same graph, different features). Model never sees reference features.

### Test Plan

**Unit tests:**
1. `test_importance_weights_shape` — weights have shape [B], all positive
2. `test_importance_weights_inversely_proportional` — high-energy bands get lower weights
3. `test_shaped_noise_unit_variance_per_band` — critical normalization check
4. `test_shaped_noise_preserves_total_variance` — total variance approximately preserved
5. `test_temporal_ramp_at_zero` — below t_knee, output equals input
6. `test_temporal_ramp_at_one` — above t_knee, output is fully shaped
7. `test_temporal_ramp_interpolation` — intermediate t gives a blend
8. `test_score_network_output_shape` — output matches input shape
9. `test_vpsde_marginal_at_zero` — mean_coeff=1, std=0
10. `test_vpsde_marginal_at_T` — x_t ≈ standard Gaussian
11. `test_vpsde_perturb_shape` — shape preserved
12. `test_training_loss_decreases` — 50 epochs on tiny graph
13. `test_generation_shape` — correct output shape

**Integration tests:**
14. `test_uniform_vs_spectral_pipeline` — end-to-end on n=30 SBM
15. `test_h2_grid_search_runs` — t_knee grid with 2 values completes

### Data Leakage Risks

1. **Train/eval feature overlap:** MITIGATED — different seeds (S vs S+1000)
2. **Importance weights from eval data:** MITIGATED — weights from Phase 0 profiling only
3. **Hyperparameter selection on test data:** MITIGATED — report full grid, H2 tests correlation not threshold
4. **Band decomposition shared:** INTENTIONAL — same bands for shaping and QBE evaluation; also report HKS and JSD as independent checks

### Run Configuration

```bash
cd ~/projects/graph-fans && uv run python -m graph_fans.phase2 \
    --output-dir results/phase2 \
    --n-nodes 200 --n-features 16 --bands 8 \
    --seeds 5 --epochs 500 \
    --t-knee-values 0.05,0.10,0.15,0.20,0.30
```

**Expected runtime:** ~3h CPU (synthetic), ~45min with GPU (Cora)

**Output artifacts:**
- `results/phase2/h1a_results.csv`
- `results/phase2/h2_results.csv`
- `results/phase2/g2_decision.json`
- `results/phase2/*.png` + `*.md`

### Success Criteria (G2 Gate)

**H1-A:**
- QBE distance (spectral) < QBE distance (uniform) in high-eigenvalue bands (5-8) on ≥2/3 families
- Paired t-test p < 0.0167 (Bonferroni-corrected for 3 families)
- Overall QBE should not increase >10%

**H2:**
- Spearman ρ > 0 between optimal t_knee* and λ₂/λ_max
- p < 0.05, bootstrap 95% CI (B=10000)
- Note: only 5 families → requires ρ ≥ 0.9 for significance

### Analysis Plan

**Plots:** Per-band comparison (H1-A), t_knee curves (H2), spectral gap correlation (H2), training curves, G2 summary
**Tables:** H1-A metrics table (mean±std), H2 grid results, statistical tests
**Statistics:** Paired t-test with Bonferroni (H1-A), Spearman + bootstrap CI (H2)
