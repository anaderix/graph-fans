# Noise Schedule Analysis: Lessons from Dieleman (2024) Applied to Graph-FANS

**Date:** 2026-03-25
**Context:** Phase 2f intermediate results show loss floor ~0.4 across all families at both scales. Model explains only 57–60% of noise variance, causing DDIM generation to produce blown-up outputs. Analysis prompted by https://sander.ai/2024/06/14/noise-schedules.html

## Core Insight

The noise schedule is a superfluous abstraction. What actually controls training is the interaction of four factors:

1. **Model parameterization** (what the model predicts → implicit SNR weighting)
2. **Explicit loss weighting** w(t)
3. **Time-step sampling distribution** p(t)
4. **Noise schedule** σ(t) (just a nonlinear reparameterization of the above)

Our current setup:
- ε-prediction → flat implicit weighting (all noise levels weighted equally in expectation)
- Uniform p(t) on [1e-5, 1.0] → most training gradient comes from **high-noise** timesteps where signal is nearly destroyed
- No explicit loss weighting w(t) = 1
- Cosine schedule σ(t)

## Diagnosis: Why Loss Floors at 0.4

With uniform t-sampling + ε-prediction, the gradient budget is misallocated:

| Noise regime | t range | SNR | What model learns | Gradient share |
|-------------|---------|-----|-------------------|----------------|
| High noise | 0.7–1.0 | <1 | "Predict Gaussian noise" | ~30% |
| Mid noise | 0.3–0.7 | 1–10 | **Spectral structure preservation** | ~40% |
| Low noise | 0.0–0.3 | >10 | Fine detail refinement | ~30% |

The high-noise regime dominates but teaches almost nothing useful — the optimal prediction there is close to the marginal mean. The mid-noise regime is where spectral structure is actually learnable (signal and noise are comparable), but it gets only ~40% of gradient budget.

The loss floor of 0.4 may reflect the model being **well-trained on the wrong noise levels**, not a fundamental capacity limit.

## Why t_knee Failed (Deeper Reason)

The t_knee mechanism varies noise *shape* (uniform → spectrally shaped) across timesteps. But Dieleman's framework reveals the real lever is *how much each timestep contributes to training*. Changing noise shape at a given timestep doesn't change how much gradient that timestep produces — it only changes what the noise looks like, not how the model is trained to handle it.

The FANS mechanism and the weighting/sampling mechanism are orthogonal:
- FANS: *what* the noise looks like at each timestep
- Weighting: *how much* each timestep matters for training

We optimized the wrong axis.

## Three Proposed Fixes (Orthogonal, Stackable)

### Fix A: Log-SNR Uniform Sampling

Replace `np.random.uniform(1e-5, T)` with uniform sampling in log-SNR space:

```python
# Current: uniform in t
t = np.random.uniform(1e-5, self.sde.T)

# Proposed: uniform in log-SNR
log_snr_min = np.log(self.sde.alpha_bar(self.sde.T) / (1 - self.sde.alpha_bar(self.sde.T)))
log_snr_max = np.log(self.sde.alpha_bar(1e-5) / (1 - self.sde.alpha_bar(1e-5)))
log_snr = np.random.uniform(log_snr_min, log_snr_max)
# Invert to get t (binary search or lookup table)
```

**Effect:** Equal gradient contribution from each SNR decade. Upweights mid-noise regime where spectral structure is learnable.

### Fix B: v-Prediction

Change model target from ε to v = α·ε − σ·x₀:

```python
# Current: ε-prediction
target = noise

# Proposed: v-prediction
mean_coeff, std = self.sde.marginal_params(t)
target = mean_coeff * noise - std * x_0
```

**Effect:** Implicit weighting becomes SNR/(1+SNR), which naturally balances low-noise and high-noise regimes. The model learns to predict the "velocity" of the diffusion process rather than the noise itself.

DDIM step for v-prediction:
```python
# x_0_hat from v-prediction:
x_0_hat = sqrt(ab_now) * x_t - sqrt(1 - ab_now) * v_pred
```

### Fix C: Min-SNR-γ Weighting (Hang et al., 2023)

Add explicit per-timestep loss weighting:

```python
snr = mean_coeff**2 / max(std**2, 1e-8)
weight = min(snr, gamma) / snr  # gamma=5 is standard
loss = weight * nn.functional.mse_loss(eps_pred, target)
```

**Effect:** Clips the loss contribution from high-SNR (low-noise) timesteps where the model can trivially predict. Upweights the mid-to-high noise regime where learning is hardest but most informative.

## Expected Impact on Loss Floor

| Fix | Mechanism | Expected loss improvement | Confidence |
|-----|-----------|--------------------------|------------|
| A (log-SNR sampling) | Rebalance gradient to mid-noise | 0.40 → 0.25–0.30 | Medium |
| B (v-prediction) | Natural SNR-balanced weighting | 0.40 → 0.25–0.35 | Medium-high |
| C (min-SNR-γ) | Clip trivial low-noise gradients | 0.40 → 0.30–0.35 | Medium |
| A+B combined | Complementary rebalancing | 0.40 → 0.15–0.25 | Medium |

If loss reaches 0.15–0.25 (75–85% noise explained), DDIM generation quality should improve dramatically — the error compounding that currently destroys outputs would be reduced by 2–4×.

## Relevance to Spectral Shaping

The W1 diagnostic showed spectral shaping provides 20–38% distributional improvement despite the model being fundamentally broken. If the loss floor drops to 0.2 (via fixes above), the model enters a regime where:

1. Generated features actually resemble training data (std ratio near 1×)
2. The 20–38% W1 advantage from spectral shaping becomes practically meaningful
3. QBE might finally show a significant difference between uniform and spectral

In other words: **spectral shaping may have been working all along, but the model was too weak to surface the benefit.**

## TODO

- [x] **NS-A**: Implement log-SNR uniform sampling in `trainer.py` — **Done**. Improves high-noise bins 3–15× but low-noise bins unchanged (0.72→0.74). Gradient rebalancing alone insufficient. #graph-fans
- [ ] **NS-B**: Implement v-prediction in `trainer.py` + `sde.py` DDIM step — **Medium** (40 lines). Natural SNR-balanced weighting via parameterization change. Well-established in literature (Salimans & Ho, 2022). Changes target, DDIM formula, and Tweedie. #graph-fans
- [x] **NS-C**: Add min-SNR-γ loss weighting to `trainer.py` — **Done**. Zero effect on per-bin loss; only changes reported total via weighting. Not useful alone. #graph-fans
- [x] **NS-D**: Run diagnostic: loss-vs-SNR profile — **Done**. Confirmed gradient misallocation: loss ranges 80× across bins, low-noise bins (7–9) at loss 0.72–1.11 with only 14% gradient. #graph-fans
- [ ] **NS-E**: Update `spectral_loss.py` Tweedie for v-prediction — **Easy** (5 lines). Required if NS-B is implemented, otherwise Tweedie formula is wrong. #graph-fans
- [ ] **NS-F**: Run W1 diagnostic with best fix combination on SBM(q=0.01) — **Easy** (reuse existing script). Compare W1 before/after to quantify whether lower loss floor makes spectral shaping benefit practically meaningful. #graph-fans
- [ ] **NS-G**: Investigate adaptive sampling schedule (EDM2-style) — **Hard** (100+ lines). Track per-SNR-bin loss with EMA, dynamically adjust sampling distribution. More principled than fixed log-SNR but higher implementation cost. Only if A+B don't break the loss floor. #graph-fans
