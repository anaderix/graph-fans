# Phase 4: Information-Guided Spectral Diffusion

Motivated by two recent papers:
- **InfoNoise** (Raya et al., arXiv:2602.18647): Information-guided noise allocation for efficient diffusion training
- **Spectrally-Guided Schedules** (Esteves & Makadia, arXiv:2603.19222): Per-instance noise schedules based on spectral properties

## Context

Phase 2g established that band-mismatched spectral noise shaping improves graph feature generation by 6-12% W1 (significant in 3/4 families after power boost). Phases 3a-3b showed that per-mode shaping and matched generation noise are counterproductive. The effect is real but modest, and downstream task impact is negligible (<2% accuracy).

The NS-D diagnostic (Phase 2f) revealed gradient misallocation: 54% of gradient goes to mid-noise bins, only 14% to the underfit low-noise regime (loss 0.72-1.10). NS-A (log-SNR sampling) and NS-C (min-SNR-gamma) failed to fix the low-noise bins.

Both papers provide new theoretical and practical tools to address these limitations.

---

## Phase 4a: InfoNoise-Guided Training (1 week)

### Motivation

InfoNoise provides a principled, data-adaptive solution to the gradient misallocation problem. It estimates the conditional entropy rate d/d-sigma H[x0|x_sigma] = mmse(sigma)/sigma^3 online from per-noise denoising losses, then concentrates training on the "informative window" where uncertainty collapses fastest.

**Why this is different from NS-A/NS-C:**
- NS-A spreads uniformly in log-SNR space (data-agnostic)
- NS-C clips loss weights at high SNR (doesn't change sampling)
- InfoNoise adaptively concentrates on the empirical entropy-rate peak (data-dependent)

**Key reframe:** Maybe the GCN doesn't NEED to learn at low noise. For community features, the "decision window" (resolving community membership) is likely in the mid-noise range where the GCN is competent (loss 0.14-0.34). InfoNoise would focus training exactly there.

### Method

1. Implement online entropy-rate estimation in `trainer.py`:
   - At each SGD step, record (sigma, loss) pair
   - Route to log-SNR bin, maintain FIFO buffer per bin (capacity B, EMA smoothing)
   - Compute r_hat(sigma) = mmse_hat(sigma) / sigma^3 per bin
   - Apply gated regularization near sigma_min to suppress boundary effects
2. Build adaptive sampling schedule pi(sigma):
   - Normalize r_hat into target density rho(sigma)
   - Convert to CDF u(sigma) for inverse-CDF sampling
   - Warm-up period: use baseline uniform sampler for N_warm steps
   - Refresh sampler every M steps
3. Keep training objective, architecture, and loss weighting unchanged (drop-in replacement)

### Evaluation

- Compare uniform t-sampling vs InfoNoise-adapted sampling on SBM(q=0.05) and BA(m=2), n=50, 5 seeds
- Primary metric: spectral W1
- **Diagnostic (publishable regardless):** Plot the graph-feature entropy-rate profile r(sigma) and compare to InfoNoise's published profiles for images and DNA

### Key Question

Where is the informative window for graph features with community structure? If it coincides with the mid-noise range, InfoNoise could amplify the 6-12% improvement. If it matches the cosine schedule's existing emphasis, the effect would be marginal.

### Risk: Medium

NS-A/NS-C results were discouraging, but InfoNoise is fundamentally different. The diagnostic value alone justifies the effort.

### Gate

InfoNoise W1 improvement > 0 on at least 1/2 families → proceed. Even if no improvement, the entropy-rate profile diagnostic informs all subsequent phases.

---

## Phase 4b: InfoGrid for DDIM (3-5 days)

### Motivation

InfoNoise's InfoGrid constructs a non-uniform sigma grid for inference, spacing DDIM steps proportionally to information density. Currently Graph-FANS uses 200 uniform DDIM steps. Many of these steps may fall in uninformative regimes (extreme high noise, or the low-noise regime where model loss ~ 1.0).

### Method

1. After training, estimate entropy-rate profile from per-sigma losses (reuse 4a infrastructure)
2. Build cumulative information coordinate: u(sigma) = integral of r_hat(sigma') d-sigma'
3. Construct non-uniform sigma grid: uniform spacing in u-space maps to non-uniform spacing in sigma-space
4. More DDIM steps in the informative window, fewer in uninformative regimes

### Evaluation

- 200 uniform DDIM steps vs 200 InfoGrid steps (same total steps, different allocation)
- Also test: can 50-100 InfoGrid steps match 200 uniform? (efficiency gain)
- Both with uniform and spectral-shaped training

### Risk: Low

Inference-only change, no retraining needed. Worst case: no difference. Best case: amplifies shaping effect by concentrating DDIM steps where spectral structure is resolved.

---

## Phase 4c: Per-Band Noise Schedules (2-3 weeks)

### Motivation

Esteves & Makadia showed that different image frequencies benefit from different noise schedules (different sigma_min, sigma_max, and temporal trajectories). For graph features, eigenmodes with different energy levels genuinely need different corruption levels:
- Band 0 (community structure): ~37-80% energy, needs high sigma_max to fully corrupt
- Bands 5-7 (high-frequency detail): ~4-6% energy, destroyed almost immediately

Currently Graph-FANS shapes noise amplitude per band at a shared timestep. Per-band schedules change the temporal dynamics: at any timestep t, different bands are at different stages of corruption.

### Key Difference from Phase 3a (which failed)

Phase 3a modulated noise amplitude at the same timestep, creating train/generate mismatch because x_T was always isotropic. Per-band schedules change TEMPORAL dynamics — at t=T, all bands are fully noised (sigma_b(T)/alpha_b(T) -> infinity for all b), so x_T remains isotropic. No mismatch at endpoints. The difference is in the intermediate trajectory.

### Method

**4c-lite: Per-band schedules (B=8 bands)**

1. For each band b, derive schedule parameters from band energy E_b:
   - sigma_max(b) proportional to E_b^{1/2} (high-energy bands need more corruption)
   - sigma_min(b) adjusted so all bands are fully noised at t=T and clean at t=0
   - Interpolation: log-linear between sigma_min(b) and sigma_max(b)
2. Forward process: c_b(t) = alpha_b(t) * c_b(0) + sigma_b(t) * epsilon_b
3. Modify SDE to support per-band alpha_bar(t, b) and sigma(t, b)
4. Modify DDIM to perform per-band reverse steps with band-specific schedule progression
5. Score network predicts per-band noise epsilon_b at the band's effective noise level

**4c-full: Per-mode schedules (n=50 modes)** — only if 4c-lite shows promise

### Schedule Derivation (following Esteves & Makadia)

For band b with energy E_b:
- kappa_max(b) = C * E_b (energy-scaled maximum noise)
- kappa_min(b) = kappa_min_base (shared minimum, set by highest-frequency band)
- Schedule: log-linear interpolation kappa_b(t) = kappa_max(b)^{1-t} * kappa_min(b)^t

Two variants (from Esteves & Makadia):
- **Power-focused:** Use E_b as probability distribution to allocate schedule bandwidth
- **Mixed:** Average of frequency-uniform and power-focused schedules

### Evaluation

- 3-way comparison: uniform vs band-amplitude-shaped (Phase 2g) vs per-band-schedule
- SBM(q=0.05), SBM(q=0.1), BA(m=2), BA(m=5), n=50, 5 seeds
- W1 metric with paired t-test

### Risk: High

Requires non-trivial changes to SDE, trainer, and DDIM. The per-band schedule creates a multi-dimensional forward process that's harder to reverse. But theoretical motivation is strong.

### Gate

Per-band schedule W1 < band-amplitude-shaped W1 by >= 5% on >= 2/4 families.

---

## Phase 4d: Spectral Conditioning of Score Network (1 week)

### Motivation

Esteves & Makadia condition the denoiser on schedule parameters: c = (y, lambda_M(t), lambda_M(0), lambda_M(1)), giving the model explicit spectral information. Currently the GCN has NO explicit access to the graph's spectral profile of features — only implicit access through the graph structure.

### Method

1. Compute spectral features from training data:
   - Band energy vector E = [E_1, ..., E_B] (B=8)
   - Importance weights g = [g_1, ..., g_B]
   - Spectral gap lambda_2
   - Energy ratio E_max / E_min
2. Concatenate [E, g, spectral_gap, energy_ratio] to the timestep embedding in the GCN
3. This gives the score network explicit knowledge of the spectral structure it should preserve
4. Train with spectral conditioning on all 4 validated families

### Evaluation

- Compare conditioned vs unconditioned GCN, both with band shaping
- Also test multi-family training: one model for all families with spectral conditioning
- W1 metric, 5 seeds per family

### Secondary Benefit

Enables cross-family transfer. One model trained on SBM+BA with spectral conditioning could generalize to unseen families.

### Risk: Low

Straightforward architecture modification. Worst case: conditioning is ignored by the model.

---

## Execution Order

| Priority | Phase | Effort | Expected Impact | Risk |
|----------|-------|--------|----------------|------|
| 1 | 4a (InfoNoise training) | 1 week | Medium-High | Medium |
| 2 | 4b (InfoGrid DDIM) | 3-5 days | Medium | Low |
| 3 | 4d (spectral conditioning) | 1 week | Medium | Low |
| 4 | 4c-lite (per-band schedules) | 2-3 weeks | High | High |

Start with 4a+4b together (2 weeks total): complementary training + inference optimization. Then 4d as a low-risk amplifier. 4c-lite only if 4a shows that per-band temporal dynamics matter more than per-band amplitude.

## Paper Implications

Even if no phase dramatically improves W1, the entropy-rate diagnostic for graph diffusion (4a) is a novel contribution — nobody has characterized the informative window for graph feature diffusion. Combined with the existing Phase 2g results, the paper story strengthens from "shaping works under specific conditions" to "information-guided spectral diffusion — a principled framework for matching noise schedules to graph spectral structure."
