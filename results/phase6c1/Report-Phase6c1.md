# Phase 6c-1: H2 + Power-Boosted 6c — Report

## Summary

Both gates failed. Band-shaped noise does not improve spectral-Wasserstein W1 over uniform noise on Cora's topology, whether the features are synthetic community-boundary features (Part A, H2) or real PCA-reduced Cora features with spectral augmentation (Part B, power-boosted 6c). The direction of effect is **negative** in both experiments.

- Part A: 3/10 subgraphs favoured band (7/10 favoured uniform).
- Part B: 10/30 subgraphs favoured band (20/30 favoured uniform). More seeds (vs Phase 6c's ~21) did not reveal a hidden positive effect; they revealed a strongly negative one.

Total GPU time: ~2h 29min (8909 s) on NVIDIA L40S. 80 training runs at ~220 s each.

## 2x2 Outcome Interpretation

| H2 | Power-6c | Paper framing |
|----|---------|---------------|
| GO | GO | Framing A works broadly. |
| GO | NO-GO | Cora's real features are the problem. |
| NO-GO | GO | Topology matters; Cora-specific. |
| **NO-GO** | **NO-GO** | **Phase 2g effect is narrow (SBM-specific topology). Framing B primary.** |

Outcome: **NO-GO + NO-GO**.

## Results Table

| Experiment | n | Uniform W1 | Band W1 | Delta % | p (band < uni) | Gate |
|------------|---|-----------:|--------:|--------:|--------------:|------|
| Part A (H2: synthetic community features on Cora topology) | 10 | 1101.4 +/- 369.4 | 1197.4 +/- 428.9 | **-8.7%** | 0.898 | **NO-GO** |
| Part B (power-boosted 6c: real features + specaug, 30 subgraphs) | 30 | 67.1 +/- 24.8 | 76.8 +/- 25.5 | **-14.4%** | 0.998 | **NO-GO** |

## Part A (H2)

Independent community-boundary features (Phase 2g protocol) on 10 Cora BFS subgraphs at n=100, d=4. Per-subgraph deltas: -210.9, +375.2, -168.3, +93.3, +495.9, +143.0, +134.7, +71.7, +118.1, -92.3. Mean +96.0. Energy ratios 3.7-13.5 (non-trivial), yet band shaping systematically underperforms. **Gate A: NO-GO** (p=0.898).

## Part B (power-boosted 6c)

30 Cora BFS subgraphs (seeds 1000-1029, disjoint from Part A), real features PCA/TruncatedSVD to d=16 (15.2% variance), spectral augmentation sigma=0.3. 100 train / 50 ref samples per subgraph (separate aug seeds). Mean delta +9.65, std 16.4 (very consistent), 10/30 favour band. **Gate B: NO-GO** (p=0.998, t=+3.20 favouring uniform). Compared to Phase 6c's null (+3 mean, p=0.748, 21 subgraphs), Phase 6c-1 Part B finds a larger, significant negative effect at 30 subgraphs — the power boost strengthens rather than overturns Phase 6c.

## Why Phase 6f showed +6.3% but Part B shows -14.4%

Phase 6f's cell D (specaug/specaug) on SBM(q=0.05) showed +6.3% with 10 seeds. Same protocol on Cora subgraphs shows -14.4%. The critical difference is graph topology. SBM(q=0.05) has dense intra-community edges with a clean gap in the Laplacian spectrum; Cora BFS subgraphs at n=100 (~170 edges, irregular structure) do not have that alignment, so band partitioning does not align with feature energy concentrations in a useful way.

## Why Part A also failed

Part A was the cleanest H2 test: Phase 2g's exact feature generator and protocol on Cora topology. It fails because Louvain communities on a 100-node Cora subgraph are much less clean than SBM's planted ones — the generated features inherit that weakness, and the eigenbasis's low band no longer cleanly encodes community identity.

## Recommendation

- **Primary (Framing B):** Spectral augmentation (Phase 6b: +46% W1 on Cora) is the practical contribution. Independent of shaping, reproducible, works on real data.
- **Secondary (Framing A, scoped):** Band noise shaping works on graphs with clean block/community topology. Positive evidence: Phase 2g (+12.5%), Phase 6f (+11.7%/+10.9%/+6.3% across 3 cells). Negative evidence: Phase 6c-1 Part A + B on Cora. Frame as a characterised result with clear scope.

## Verdict

- Phase 6c-1 Part A (H2): **NO-GO**
- Phase 6c-1 Part B (power-boosted 6c): **NO-GO**
- Phase 6c's original NO-GO stands and strengthens.
- Paper framing: **Framing B primary**, Framing A as scope-bounded secondary.

## Files

- Script: `scripts/test_phase6c1_h2_plus_power.py`
- Results JSON: `results/phase6c1/h2_plus_power.json`
- Run log: `results/phase6c1/phase6c1_run.log`
- Tests: `tests/test_phase6.py` (class `TestPhase6c1`, 4 smoke tests, all 39 pass)
- Commits: `d532d91` (code+tests), `553be9a` (results+LOG+CLAUDE)
