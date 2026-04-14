# Phase 5: Frequency-Domain Diffusion

## Motivation

All Phase 2-4 approaches diffuse in the spatial (node feature) domain and rely on a GCN score network to implicitly learn spectral structure. The GCN's polynomial spectral response creates a fundamental expressiveness ceiling (Phase 2f diagnosis: loss 0.72-1.10 at low noise, 6-layer GCN gives only 5-13% improvement). This phase pivots to diffusing directly in the spectral domain, bypassing the GCN bottleneck entirely.

Motivated by:
- **Alt-5 from original roadmap:** Direct spectral generation, generating per-band coefficients independently
- **Esteves & Makadia (arXiv:2603.19222):** Per-frequency noise schedules derived from spectral properties
- **InfoNoise (Raya et al., arXiv:2602.18647):** Information-guided allocation naturally applies per-mode in spectral domain
- **Phase 2f architecture comparison:** Loss != generation quality; the GCN's implicit regularization helps but also limits

## Core Idea

Instead of diffusing features x in R^{n x d} and shaping noise, diffuse spectral coefficients c in R^{n x d} where c = U^T x:

1. **Project:** c_k = U_k^T x (k-th eigenmode coefficient vector, d-dimensional)
2. **Forward:** Each mode k follows its own 1D diffusion with mode-specific schedule:
   c_k(t) = alpha_k(t) * c_k(0) + sigma_k(t) * epsilon_k
3. **Score network:** Small MLP per mode, or shared MLP conditioned on eigenvalue lambda_k
4. **Reverse:** Per-mode DDIM with mode-specific schedule
5. **Reconstruct:** x = U c

### Why This Could Work

- Each mode's diffusion is a trivial denoising problem (d-dimensional Gaussian, no graph convolution needed)
- Mode-specific schedules are exact, not approximated through band averaging
- The spectral energy profile directly parameterizes each mode's schedule
- InfoNoise can be applied per-mode to find each mode's informative window
- No GCN polynomial filter bottleneck

### Why It Might Not Work

- **Cross-mode correlations:** Real features have correlated spectral coefficients. Community boundary features have correlated low/mid frequency content. Independent per-mode generation would lose these correlations.
- **Fixed topology assumption:** Requires eigendecomposition, so only works for Regime A (fixed graphs).
- **Reconstruction artifacts:** Small per-mode errors could compound when projecting back to spatial domain.

## Method

### 5a: Independent Per-Mode Diffusion (simplest, 1 week)

Each eigenmode diffused independently:
- Score network: shared 3-layer MLP(d -> 128 -> 128 -> d) conditioned on (lambda_k, t, E_k)
- Per-mode schedule: sigma_k(t) derived from mode energy E_k following Esteves & Makadia
- Per-mode DDIM with InfoGrid step allocation
- No cross-mode interaction

**Evaluation:** W1 metric (which is already per-mode, so this is a natural fit).

**Expected failure mode:** Generated features may look spectrally correct per-mode but spatially incoherent (each node's feature vector is a sum of independently generated mode contributions).

### 5b: Autoregressive Spectral Diffusion (2 weeks)

Generate modes sequentially from low to high frequency, conditioning each mode on previously generated lower modes:
- Mode 0 (lowest frequency): unconditional diffusion
- Mode k: diffusion conditioned on c_0, c_1, ..., c_{k-1}
- Captures the dominant cross-mode correlations (low-freq structure constrains high-freq detail)
- Score network: MLP conditioned on (lambda_k, t, E_k, c_{<k})

**Trade-off:** Much slower generation (sequential over n modes) but captures correlations.

**Practical variant:** Autoregressive over B=8 bands (not individual modes). Generate band 0 coefficients, then band 1 conditioned on band 0, etc. 8 sequential steps instead of n=50.

### 5c: Cross-Mode Attention Diffusion (2-3 weeks)

All modes diffused in parallel but with lightweight cross-mode attention:
- Score network: MLP per mode + cross-mode transformer attention layer
- Attention over modes at each denoising step: each mode can attend to other modes' current state
- Preserves parallelism while capturing correlations

**Architecture:**
```
for each DDIM step:
    h_k = MLP(c_k(t), lambda_k, t, E_k)  # per-mode features
    h = CrossAttention(h_1, ..., h_n)      # cross-mode interaction
    epsilon_k = ProjectionHead(h_k)         # per-mode noise prediction
```

## Schedule Derivation

For mode k with energy E_k = E[||c_k||^2]:

**From Esteves & Makadia:**
- sigma_max(k) = C * sqrt(E_k) (energy-scaled maximum noise level)
- sigma_min(k) = sigma_min_base (shared, set by reconstruction precision)
- Interpolation: cosine schedule within [sigma_min(k), sigma_max(k)]

**From InfoNoise:**
- Per-mode entropy rate r_k(sigma) = mmse_k(sigma) / sigma^3
- Per-mode informative window from online estimation
- Per-mode InfoGrid for DDIM step allocation

## Evaluation Plan

### Metrics
- **Spectral W1** (primary): natural fit since W1 is already per-mode
- **Spatial coherence:** Correlation between generated features of adjacent nodes (should be high for community features)
- **Downstream task:** Node classification accuracy
- **Reconstruction fidelity:** ||x - U U^T x|| (should be zero for exact eigenbasis, measures numerical stability)

### Baselines
1. Uniform spatial diffusion (Phase 2 baseline)
2. Band-shaped spatial diffusion (Phase 2g best)
3. Phase 4 best result (if available)

### Families and Scale
- SBM(q=0.05), SBM(q=0.1), BA(m=2), BA(m=5), n=50, 5 seeds
- If successful at n=50, test n=100, 200 (eigendecomposition is O(n^3) but amortized over training)

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Cross-mode correlations lost (5a) | High | Major | Move to 5b or 5c |
| Spatial incoherence in reconstructed features | Medium | Major | Add spatial smoothness regularization; condition on local graph structure |
| Per-mode MLP too simple for d-dim denoising | Low | Minor | Increase MLP capacity; d=4 is very low-dimensional |
| Eigendecomposition cost at larger scale | Medium | Moderate | Amortize: compute once per graph, cache. Chebyshev approximation for >500 nodes |
| Numerical instability in U^T x / U c projection | Low | Minor | Use double precision for eigendecomposition; orthogonality check |

## Connection to Roadmap

This is Alt-5 from the Phase 2 root cause analysis ("Direct spectral generation — generate in eigenbasis, per-band diffusion"). It was deprioritized then because the spatial approach hadn't been fully debugged. Now that Phases 2f-3b have established the spatial approach's ceiling (6-12% W1 from shaping, limited by GCN capacity), and both papers provide theoretical grounding for per-frequency treatment, the case for spectral-domain diffusion is much stronger.

Phase 5 is independent of Phase 4 results. If Phase 4 shows large improvements, Phase 5 becomes less urgent. If Phase 4 plateaus, Phase 5 is the natural pivot.

## Effort Estimate

| Variant | Effort | Risk | Potential |
|---------|--------|------|-----------|
| 5a (independent) | 1 week | Medium | Moderate (W1 improvement, spatial coherence concern) |
| 5b (autoregressive) | 2 weeks | Medium | High (captures correlations, slower generation) |
| 5c (cross-attention) | 2-3 weeks | High | Highest (parallel + correlations, most complex) |

Start with 5a as a diagnostic: if independent per-mode diffusion already beats spatial diffusion on W1, the spectral domain approach is validated and cross-mode modeling (5b/5c) addresses the remaining coherence gap. If 5a fails on W1, the approach may be fundamentally limited.
