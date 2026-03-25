# Phase 2g Follow-up Plan: Extending the First Significant Result

**Date:** 2026-03-25
**Context:** Phase 2g showed 12.5% W1 improvement on SBM(q=0.05) at 50 nodes (p=0.0001). This plan validates whether the effect generalizes across families and scales, and whether it matters for downstream tasks. H2 (temporal ramp / t_knee) is dropped — definitively a no-op.

## Decision: Drop H2 (t_knee)

The W1 diagnostic showed <1% variation across all t_knee values on both families. Remove from all future experiments:
- [ ] **CLEANUP-1**: Remove t_knee grid from experiment pipeline — **Easy**. Strip H2 from `run_experiment.py`, `evaluate.py`. This halves compute (no more 5×t_knee × 5×seed H2 grid). #graph-fans
- [ ] **CLEANUP-2**: Simplify `run_experiment.py` to run H1-A only — **Easy**. Drop `run_h2_experiment`, `compute_g2_decision` H2 branch. Keep G2 decision based on H1-A alone. #graph-fans

## Step 1: More Families (validate generalization)

**Goal:** Test whether the 12.5% W1 improvement generalizes beyond SBM(q=0.05).

**Families to test:** SBM(q=0.1), BA(m=2), BA(m=5) — all at 50 nodes, 4 features, 5 seeds.

**Prediction by spectral structure:**

| Family | Spectral profile | Expected shaping effect |
|--------|-----------------|----------------------|
| SBM(q=0.01) | 80% band 0 (unimodal) | None (confirmed: −4.8%, p=0.69) |
| **SBM(q=0.05)** | **39%+37% bimodal** | **YES (confirmed: +12.5%, p=0.0001)** |
| SBM(q=0.1) | 44%+27% bimodal | Likely yes (similar to q=0.05) |
| BA(m=2) | 32%+45% in bands 0–1 (adjacent) | Maybe — energy in adjacent bands, less to reshape |
| BA(m=5) | 26%+41% bimodal (bands 0+2) | Likely yes (bimodal like SBM q=0.05) |

- [ ] **FAMILY-1**: Run `test_shaping_w1.py` on SBM(q=0.1), BA(m=2), BA(m=5) — **Easy** (modify script to accept family list, or run 3× manually). ~2h on GPU. #graph-fans
- [ ] **FAMILY-2**: Analyse results — which families show significant W1 improvement? Correlate with spectral profile bimodality. #graph-fans

**Success criterion:** ≥2 of 3 new families show significant improvement (p<0.025). If so, the effect generalizes to any graph with multi-scale spectral structure.

**Estimated time:** 2–3 hours (GPU) + 30 min analysis.

## Step 2: Scale Study (the make-or-break question)

**Goal:** Test whether the effect persists, grows, or vanishes at larger graphs.

**Scales:** 50 (done), 100, 150, 200 nodes. All with SBM(q=0.05) and BA(m=5) (families most likely to show effect).

**Key challenge:** At 200 nodes the 3L GCN couldn't denoise (std ratio 8–10×). Options:
1. Scale features proportionally: keep n_features=4 at all scales (reduces dimensionality pressure)
2. Increase hidden_dim with scale: 128 @ 50n, 192 @ 100n, 256 @ 200n
3. Accept that 200 nodes may not work with 3L GCN — the scale limit IS a finding

- [ ] **SCALE-1**: Run at 100 nodes with n_features=4, hidden_dim=128, 3L GCN — **Easy**. Direct extension of 2g config. Generate datasets, train 5 seeds × 2 methods, measure W1. #graph-fans
- [ ] **SCALE-2**: Run at 150 nodes same config — **Easy**. #graph-fans
- [ ] **SCALE-3**: Run at 200 nodes same config — **Easy**. May fail (model can't denoise). That's informative. #graph-fans
- [ ] **SCALE-4**: If 200n fails with hidden_dim=128, retry with hidden_dim=256 — **Easy**. Tests whether width can compensate. #graph-fans
- [ ] **SCALE-5**: Analyse scaling curve — plot W1 improvement % vs n_nodes. Is it constant, growing, or decaying? #graph-fans

**Success criterion:** Effect persists at ≥100 nodes (doesn't vanish immediately). Effect growing with scale would be the strongest possible finding for the paper.

**Estimated time:** 4–8 hours (GPU) + 1h analysis.

## Step 3: Downstream Task Evaluation (practical significance)

**Goal:** Does 12.5% W1 improvement translate to measurable performance on a real task?

**Approach:**
1. Generate N=1000 feature matrices per method (uniform vs spectral) on SBM(q=0.05)
2. Use generated features for node classification (predict community membership)
3. Train a simple classifier (logistic regression) on generated features, evaluate on held-out real features
4. Compare classification accuracy between uniform-generated and spectral-generated features

- [ ] **TASK-1**: Implement node classification evaluation — **Medium** (new script, ~100 lines). Use sklearn logistic regression on spectral coefficients. #graph-fans
- [ ] **TASK-2**: Run on SBM(q=0.05) and BA(m=5) with both methods — **Easy** (reuse existing generation). #graph-fans
- [ ] **TASK-3**: Statistical comparison (paired t-test over seeds) — **Easy**. #graph-fans

**Success criterion:** Spectral-generated features produce ≥2% higher classification accuracy. This would demonstrate practical significance beyond distributional metrics.

**Estimated time:** 1 day implementation + 2h GPU + analysis.

## Step 4: Write Up

**Goal:** Draft the paper around the Graph-FANS result.

**Story arc:**
1. FANS works in image diffusion → does spectral noise shaping transfer to graphs?
2. 6 iterations of negative results (2a–2f) — systematic elimination of confounds
3. Discovery: loss ≠ generation quality (3L GCN beats Transformer at generation)
4. Discovery: QBE misses distributional effects → W1 as proper evaluation metric
5. Positive result: 12.5% W1 improvement on bimodal-spectrum graphs
6. Scaling and downstream validation (from Steps 1–3)

**Contributions:**
- Spectral W1 metric for evaluating graph diffusion models
- Empirical finding: spectral noise shaping helps when graph has multi-scale structure
- Negative result: temporal ramp (t_knee) has zero effect
- Insight: simpler models (GCN) generate better than complex ones (Transformer) in iterative sampling

- [ ] **PAPER-1**: Draft introduction + related work — **Medium**. #graph-fans
- [ ] **PAPER-2**: Draft experiments section with tables from 2a–2g — **Medium**. #graph-fans
- [ ] **PAPER-3**: Add results from Steps 1–3 — depends on those completing first. #graph-fans

## Execution Order

```
Week 1:
  CLEANUP-1,2 (1h) → FAMILY-1,2 (3h) → SCALE-1,2,3 (6h)
                                        → SCALE-4,5 if needed (3h)

Week 2:
  TASK-1,2,3 (1.5 days)
  PAPER-1,2 (parallel with TASK)

Week 3:
  PAPER-3 (incorporate results)
  Review and submit
```

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Effect vanishes at 100 nodes | Still publishable: "spectral shaping works at small scale" + methodological contributions |
| Only SBM(q=0.05) shows effect | Narrow but real result; focus paper on conditions for effect |
| Downstream task shows no benefit | W1 improvement is still a contribution; reframe as evaluation metric paper |
| Reviewer asks about 200+ nodes | Acknowledge as limitation; point to capacity analysis as explanation |
