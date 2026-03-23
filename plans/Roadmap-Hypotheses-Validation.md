---
tags: [project, graph-fans, roadmap, planning]
created: 2026-03-22
status: active
---

# Graph-FANS: Hypotheses Validation Roadmap

Detailed phased roadmap for validating hypotheses H1–H6 from [[Scale-Aware Training Schedules for Diffusion Networks]].

## Dependency Graph

```
Phase 0 (Step 0: Spectral Profiling)
  │
  ├──► Phase 1a (H3: Spectral Metrics)
  │       │
  │       ▼
  ├──► Phase 2 (H1-A + H2: Regime A Core)
  │       │
  │       ▼
  │    Phase 3 (H4: Persistence-Informed Bands)
  │       │
  │       ▼
  │    Phase 4 (H5: Hyperbolic Proxy)
  │
  └──► Phase 1b (Synthetic Benchmarks)
          │
          ▼
       Phase 5 (H1-B + H6: Regime B + Point Clouds)
```

## Go/No-Go Gates

| Gate | After Phase | Condition to Continue | Fallback if Failed |
|------|-------------|----------------------|-------------------|
| **G0** | Phase 0 | Graph spectral energy shows non-uniform distribution across bands (≥2× variation between lowest and highest band energy) | Abandon inverse-power weighting; explore data-driven weighting or pivot to different noise shaping strategy |
| **G1** | Phase 1a | At least 1 of 3 proposed metrics (spectral density JSD, quantile-band energy, HKS distance) shows ≥10% relative difference from aggregate MMD on existing baselines | H3 falsified — spectral evaluation adds no signal beyond existing metrics; reconsider whether spectral bias is a real bottleneck |
| **G2** | Phase 2 | Graph-FANS Regime A improves spectral fidelity over uniform noise baseline on ≥2 of 3 tested graph families | Core thesis unsupported; write up negative result and stop |
| **G3** | Phase 3 | Persistence-informed bands outperform uniform bands on irregular graphs (CV > 1.0) by ≥5% spectral fidelity | H4 rejected — uniform/quantile bands sufficient; proceed to Phase 4 without persistence |
| **G4** | Phase 4 | Hyperbolic curvature proxy achieves ≥80% of full eigendecomposition fidelity gain on tree-like graphs | H5 rejected — no scalable proxy available; Regime B requires explicit spectral methods |

---

## Phase 0 — Empirical Spectral Profiling (Step 0)

**Goal:** Determine whether graph signal energy follows a non-uniform spectral profile amenable to FANS-style importance weighting.

**Duration:** 2–3 weeks

**Hypotheses gated:** All (H1–H6) — this is the empirical prerequisite.

### Tasks

- [ ] Select 3+ graph families for profiling:
  - (a) Stochastic Block Models (SBM) with varying modularity $q \in \{0.01, 0.05, 0.1, 0.2\}$
  - (b) Barabási-Albert preferential attachment ($m \in \{2, 5, 10\}$)
  - (c) Citation networks (Cora, CiteSeer)
  - (d) Optional: molecular graphs (QM9 subset), protein-protein interaction
- [ ] For each graph family: compute Laplacian eigendecomposition, partition into $B=8$ bands, measure per-band signal energy on node features
- [ ] Characterize the spectral energy distribution:
  - Does it decay (power-law, exponential)?
  - Is it concentrated in specific band ranges?
  - How much does it vary across graph families?
- [ ] Determine appropriate weighting scheme:
  - If decaying: inverse-power weighting (FANS-style) is appropriate → proceed
  - If flat: data-driven per-family weighting needed
  - If concentrated: targeted band boosting rather than global scheme
- [ ] Document results as spectral energy profiles (plots + statistics)

### Deliverables

- Spectral energy profile plots for each graph family
- Statistical summary: band energy ratios, decay rates, cross-family comparison
- **Go/No-Go decision (G0):** Is non-uniform noise shaping justified?

### Success Criteria

- Energy ratio between lowest and highest band ≥2× for at least 2 graph families
- Clear pattern (decaying, concentrated, or structured) — not noise

---

## Phase 1a — Basis-Free Spectral Metrics (H3)

**Goal:** Implement and validate basis-free spectral evaluation metrics that reveal fidelity gaps invisible to standard metrics.

**Duration:** 2–3 weeks (can start immediately after G0, parallel with Phase 1b)

**Hypothesis:** H3 — Scale-Dependent Evaluation Gap

### Tasks

- [ ] Implement 3 basis-free metrics:
  - **Spectral density distance:** KDE on normalized eigenvalues $\lambda_i / \lambda_{\max}$, compute JSD between generated and reference
  - **Quantile-band energy:** Equal-mass quantile bands, per-band energy comparison
  - **Heat kernel signature distance:** $\text{Tr}(e^{-tL})$ over logarithmic $t$ grid
- [ ] Validate metrics on known distributions:
  - Generate graphs from GDSS and/or SPECTRE baselines
  - Compute both standard metrics (MMD on degree/clustering/orbit) and proposed spectral metrics
  - Quantify the information gap: do spectral metrics reveal degradation that MMD misses?
- [ ] Test falsification criterion: spectral density JSD shows ≥10% relative difference from aggregate MMD metrics
- [ ] Benchmark computational cost of each metric

### Deliverables

- Metric implementation (reusable evaluation library)
- Comparison table: standard vs spectral metrics on baseline-generated graphs
- **Go/No-Go decision (G1):** Do spectral metrics add diagnostic value?

### Success Criteria

- At least 1 metric reveals ≥10% relative difference from aggregate MMD (H3 not falsified)
- Metrics are computationally feasible for evaluation loops

---

## Phase 1b — Synthetic Benchmarks

**Goal:** Build synthetic graph benchmarks with known multiscale structure for controlled hypothesis testing.

**Duration:** 2 weeks (parallel with Phase 1a)

### Tasks

- [ ] Design hierarchical SBM benchmarks (analogous to FANS's PLTB/EGM):
  - 2-level hierarchy: communities within communities
  - Known spectral structure: tunable inter/intra-community edge probabilities
  - Ground-truth band energy distribution for validation
- [ ] Implement benchmark generation code with configurable parameters:
  - Number of levels, community sizes, edge probability ratios
  - Node feature distributions (smooth vs rough across communities)
- [ ] Generate training/validation/test splits for 3 difficulty levels
- [ ] Verify that spectral metrics (from Phase 1a) correctly capture known structure

### Deliverables

- Synthetic benchmark generation code
- 3 benchmark datasets with documented ground-truth spectral properties
- Verification that Phase 1a metrics align with ground truth

---

## Phase 2 — Regime A Core (H1-A + H2)

**Goal:** Implement and validate Graph-FANS noise shaping on fixed-topology graphs with temporal ramping.

**Duration:** 4–6 weeks

**Hypotheses:** H1 (Regime A), H2 (Temporal Ramp)

**Prerequisites:** G0 passed, G1 passed, Phase 1b complete

### Tasks

#### H1-A: Spectral Band Decomposition (Regime A)

- [ ] Implement Laplacian eigenvalue band decomposition for fixed graphs
- [ ] Implement FANS-style importance profiling on graph signals:
  - Per-band energy computation on node features
  - Inverse-power weight derivation $g_b \propto (\bar{\pi}_b + \epsilon)^{-\alpha}$
  - Per-eigenband noise normalization (unit variance per band — critical for stability)
- [ ] Integrate with a graph diffusion backbone (GDSS or equivalent):
  - Replace uniform Gaussian noise with spectrally-shaped noise
  - Maintain compatibility with score-based SDE framework
- [ ] Train on: (a) synthetic benchmarks, (b) SBMs, (c) Cora, (d) QM9 subset
- [ ] Evaluate with Phase 1a spectral metrics + standard metrics (MMD)
- [ ] Compare against uniform-noise baseline:
  - Per-band spectral fidelity (expecting high-eigenvalue improvement)
  - Overall generation quality (MMD, validity for molecules)

#### H2: Hierarchical Temporal Ramp

- [ ] Implement temporal ramp $\phi(t)$: low-eigenvalue → high-eigenvalue noise allocation
- [ ] Implement white noise guardrail below $t_{\text{knee}}$
- [ ] Grid search: $t_{\text{knee}} \in \{0.05, 0.10, 0.15, 0.20, 0.30\}$ with fixed $\beta, \gamma, \alpha, B$
- [ ] Test across 5 graph families: SBMs (4 modularity levels), Erdős-Rényi, Barabási-Albert, Cora, QM9
- [ ] Measure Spearman rank correlation between optimal $t_{\text{knee}}$ and spectral gap ratio $\lambda_2/\lambda_{\max}$
- [ ] Report bootstrap 95% CIs

### Deliverables

- Graph-FANS Regime A implementation
- Comparison tables: Graph-FANS vs uniform noise across graph families
- H2 validation: $t_{\text{knee}}$ vs spectral gap correlation plot with CIs
- **Go/No-Go decision (G2):** Does spectral noise shaping improve graph generation?

### Success Criteria

- H1-A: Spectral fidelity improvement in high-eigenvalue bands on ≥2 of 3 graph families
- H2: Statistically significant correlation ($p < 0.05$) between optimal $t_{\text{knee}}$ and $\lambda_2/\lambda_{\max}$

---

## Phase 3 — Band Optimization (H4)

**Goal:** Test whether persistence-informed band boundaries outperform uniform/quantile partitioning.

**Duration:** 3–4 weeks

**Hypothesis:** H4 — Persistence-Informed Band Boundaries

**Prerequisites:** G2 passed (Regime A works)

### Tasks

- [ ] Implement persistent homology computation on graph filtrations (Vietoris-Rips on shortest-path distances)
- [ ] Extract persistence diagram birth-death pairs; identify natural scale boundaries (gaps in persistence)
- [ ] Compute hierarchical irregularity metric: CV of persistence lifetimes ($H_0 + H_1$)
- [ ] Classify test graphs: irregular (CV > 1.0) vs regular (CV < 0.5)
- [ ] Run 4-condition ablation:
  - (i) Uniform $\lambda$-bands
  - (ii) Quantile $\lambda$-bands
  - (iii) Persistence-informed bands
  - (iv) Random bands (matched count $B$)
- [ ] Evaluate on irregular graphs: Facebook social, protein-protein, Cora
- [ ] Evaluate on regular graphs: uniform SBMs, lattice graphs
- [ ] Statistical analysis: $n=5$ seeds, mean ± std, paired t-test with Bonferroni correction
- [ ] Test cross-dataset transfer of persistence-informed bands

### Deliverables

- Persistence-informed band computation pipeline
- 4-condition ablation results table with statistical tests
- Analysis of irregularity–advantage correlation
- **Go/No-Go decision (G3):** Are persistence-informed bands worth the computational overhead?

### Success Criteria

- Persistence-informed bands outperform uniform bands by ≥5% spectral fidelity on irregular graphs (CV > 1.0)
- Advantage is significantly larger on irregular vs regular graphs

---

## Phase 4 — Scalability Proxy (H5)

**Goal:** Validate hyperbolic embedding curvature as a cheap proxy for spectral band importance weights.

**Duration:** 3–4 weeks

**Hypothesis:** H5 — Hyperbolic Geometry as Spectral Prior

**Prerequisites:** G2 passed

### Tasks

- [ ] Implement Poincaré ball embeddings for test graphs (use existing library: geoopt or similar)
- [ ] Derive band importance weights from hyperbolic curvature:
  - Map node depth in hyperbolic space to eigenvalue band assignment
  - Compute curvature-derived weights as proxy for inverse-power weights
- [ ] Compare curvature-derived weights against full eigendecomposition weights:
  - Weight vector correlation (Pearson/Spearman)
  - Spectral fidelity gap: Graph-FANS with curvature proxy vs Graph-FANS with exact weights
- [ ] Test on 3 graph families:
  - (a) Trees/DAGs — expect strong correspondence
  - (b) Hierarchical networks with moderate cross-links — expect moderate
  - (c) Dense networks — expect poor correspondence (known limitation)
- [ ] Benchmark computational cost: $O(n \log n)$ proxy vs $O(n^3)$ eigendecomposition
- [ ] Identify the "crossover" graph size where proxy becomes essential

### Deliverables

- Curvature-to-band-importance mapping implementation
- Correlation analysis: curvature weights vs eigendecomposition weights
- Fidelity comparison table across graph families
- Computational cost benchmarks
- **Go/No-Go decision (G4):** Is the proxy viable for large-scale graphs?

### Success Criteria

- ≥80% of full eigendecomposition fidelity gain on tree-like graphs
- Clear computational advantage at graph sizes >10K nodes

---

## Phase 5 — Regime B & Point Clouds (H1-B + H6)

**Goal:** Extend Graph-FANS to topology generation (Regime B) and point cloud generation (H6).

**Duration:** 6–8 weeks

**Hypotheses:** H1 (Regime B), H6

**Prerequisites:** G2 passed, ideally G3 and G4 results available to inform design choices

### Tasks

#### H1-B: Basis-Free Noise Shaping (Regime B)

- [ ] Implement Chebyshev polynomial band-pass filters $p_b(L)$:
  - Approximate band-pass filtering without eigendecomposition
  - Validate filter response against exact spectral decomposition
- [ ] Implement spectral density matching during generation:
  - Profile target distribution $\rho(\lambda)$ via KDE on training set eigenvalue histograms
  - Shape noise to match target spectral density
- [ ] Integrate with topology generation pipeline:
  - Adjacency matrix diffusion with spectrally-shaped noise
  - Per-step Chebyshev filter application (no eigenvector computation needed)
- [ ] Evaluate on graph generation benchmarks:
  - Community graphs, molecular graphs, social network subgraphs
  - Compare against GDSS, SPECTRE baselines

#### H6: Point Cloud Generation

- [ ] Implement HNSW-induced graph construction from noisy point clouds
- [ ] Implement fixed-initialization strategy:
  - Construct k-NN graph once at $t=T$ from initial noisy points
  - Keep graph fixed throughout denoising
- [ ] Implement Chebyshev filter fallback:
  - Polynomial filters $p_b(L_0)$ on initial Laplacian $L_0$
  - Compare against eigenvector-based shaping
- [ ] Optional: temporal adjacency smoothing (exponential moving average with decay $\gamma$) if fixed scaffold degrades
- [ ] Train on ModelNet10 subset
- [ ] Evaluate:
  - Micro-scale fidelity: sharp edges, surface detail
  - Heat kernel signature distance on induced graph
  - Compare against uniform-noise point cloud diffusion baseline
- [ ] Test scaffold alignment:
  - Does the fixed initial graph remain aligned with evolved points?
  - At what noise level does misalignment become problematic?

### Deliverables

- Regime B implementation with Chebyshev filters
- H6 point cloud pipeline with fixed-initialization strategy
- Evaluation results: spectral fidelity, generation quality, scaffold alignment analysis
- Failure mode analysis: when does the fixed-initialization assumption break?

### Success Criteria

- H1-B: Regime B generates graphs with better spectral fidelity than uniform-noise baselines
- H6: Point cloud diffusion with Graph-FANS recovers higher micro-scale fidelity than uniform baseline
- Clear characterization of when fixed-initialization works vs fails

---

## Timeline Summary

| Phase | Duration | Hypotheses | Key Gate |
|-------|----------|-----------|----------|
| **Phase 0** | Weeks 1–3 | Prerequisite for all | G0: Non-uniform spectral energy? |
| **Phase 1a** | Weeks 3–5 | H3 | G1: Spectral metrics informative? |
| **Phase 1b** | Weeks 3–5 | Infrastructure | — |
| **Phase 2** | Weeks 5–11 | H1-A, H2 | G2: Regime A works? |
| **Phase 3** | Weeks 11–15 | H4 | G3: Persistence helps? |
| **Phase 4** | Weeks 11–15 | H5 | G4: Proxy viable? |
| **Phase 5** | Weeks 15–23 | H1-B, H6 | — |

Phases 3 and 4 can run in parallel after G2. Total estimated duration: ~23 weeks (5–6 months) assuming serial execution of dependent phases.

## Resource Requirements

- **Compute:** GPU access for diffusion model training (Phase 2 onward). CPU sufficient for Phases 0, 1a, 1b.
- **Libraries:** PyTorch Geometric, NetworkX, giotto-tda (persistent homology), geoopt (hyperbolic embeddings), SDE solvers
- **Datasets:** SBM/BA generators (synthetic), Cora/CiteSeer (public), QM9 (public), ModelNet10 (public), Facebook/PPI (public)
- **Baselines to reproduce:** GDSS, optionally SPECTRE

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| G0 fails — spectral energy is flat | Medium | Fatal | Pivot to data-driven weighting or abandon spectral shaping thesis |
| Eigendecomposition too slow for training loop | High | Major | Chebyshev approximation (already planned for Regime B); amortize over training set |
| Normalization instability (FANS critical detail) | Medium | Major | Follow FANS unit-variance protocol; ablate normalization strategies early in Phase 2 |
| H2 correlation is weak | Medium | Minor | $t_{\text{knee}}$ can be tuned per-family without theoretical grounding; H2 is nice-to-have |
| Fixed initialization fails for H6 | Medium | Moderate | Fall back to Chebyshev filters; characterize failure regime |
| Concurrent work scoops key results | Low-Medium | Major | Focus on unique contributions (persistence-informed bands, two-regime framework) |
