# Phase 6a Code Review

**Reviewer:** Tester/Reviewer agent
**Date:** 2026-04-14
**Files reviewed:**
- `graph_fans/phase6/__init__.py`
- `graph_fans/phase6/subgraph_sampler.py`
- `graph_fans/phase6/feature_reducer.py`
- `graph_fans/phase6/augmentation.py`
- `scripts/test_phase6a_scale.py`
- `tests/test_phase6.py`

---

## 1. Data Leakage Audit

### PCA/TruncatedSVD fit scope — PASS
The reducer is fit once on all 2708 Cora nodes (`fit_feature_reducer(full_features, ...)` at line 253 of the experiment script), then reused for every subgraph via `reduce_features(sub_features_full, reducer)`. No per-subgraph fitting. Matches plan requirement exactly.

### Augmentation noise independence — PASS
`gaussian_augmentation` adds zero-mean Gaussian noise to the real features. The reference for evaluation is the un-augmented real features (`sub_features[np.newaxis, ...]`). Noise is generated from an independent RNG seeded per-subgraph. No information about the reference leaks into the augmented training data beyond the shared base features (which is the intended design — the augmentation creates variations of the same real features).

### Band importance weights — PASS (N/A for 6a)
Phase 6a uses `noise_shaping="uniform"` exclusively (line 122 of experiment script). No importance weights are computed. This is correct — Phase 6a is a scale diagnostic (baseline denoising), not a shaping test. Importance weights are deferred to Phase 6c.

---

## 2. Plan Conformance

### BFS subgraph sampling — PASS
- Returns connected induced subgraphs: BFS from a single root guarantees a connected visit order; the prefix truncation preserves connectivity. An additional safety net extracts the largest connected component.
- Relabeling: nodes are mapped to `0..n-1` via `nx.relabel_nodes` with an explicit mapping dict. Verified by `test_bfs_subgraph_relabeled`.
- Edge preservation: the induced subgraph retains only edges from the original graph. Verified by `test_bfs_subgraph_preserves_edges`.
- Accepts subgraphs with >= 80% of requested nodes (graceful degradation for small components). For Cora's largest component (2485 nodes), BFS to n=200 always succeeds in full.

### Feature reduction — PASS
- `fit_feature_reducer` supports both `"truncated_svd"` (for sparse Cora) and `"pca"` (for continuous features like PubMed TF-IDF). Both tested.
- TruncatedSVD correctly avoids centering, preserving sparsity benefits. PCA centers data, appropriate for continuous features.

### Gaussian augmentation — PASS
- Output shape: `[N, n_nodes, n_features]` from a single `[n_nodes, n_features]` input. Verified by `test_shape`.
- Implementation: `x_aug = features + sigma_i * randn(n_nodes, n_features)` where sigma_i varies per sample (drawn uniformly from a scaled range). This matches the plan's `x_aug = x_real + sigma * randn` with the improvement of per-sample sigma diversity.
- Mean preservation: E[x_aug] = x_real. Verified empirically by `test_mean_close_to_original` (500 samples, atol=0.15).

### Experiment grid — PASS
- Default scales: `"50,100,150,200"` — matches plan.
- Default dims: `"4,16,32"` — matches plan.
- Default subgraphs: 20 — matches plan.
- Default epochs: 500 — matches plan.
- Training config: 128 hidden, 3 layers, GCN, cosine SDE, EMA, LR scheduler — matches plan ("3L GCN 128h, cosine+EMA").

### Gate check — PASS
- Gate cell: n=100, d=16.
- Pass criterion: `pass_rate >= 75%` (15/20 subgraphs with std_ratio < 3.0) AND `energy_ratio >= 2.0x`.
- Matches plan exactly.

### Quick mode — BONUS
`--quick` flag reduces to 3 subgraphs and 50 epochs for rapid testing. Good ergonomics.

---

## 3. Correctness

### Subgraph node indices to feature row mapping — PASS
Line 87: `sub_features_full = full_features[sample.node_indices]` correctly indexes the full feature matrix using the original node IDs stored in `node_indices`. After reduction, `sub_features.shape == (n, d)` where n is the actual subgraph size. Verified by `test_full_pipeline_smoke` (line 283: `sub_features = reduce_features(full_features[sample.node_indices], reducer)`; asserts shape `(50, 4)`).

### Feature reducer applied to subgraph features — PASS
The reducer is fit on `full_features` (2708 nodes) but `.transform()` is called on `sub_features_full` (the subgraph's subset of the original features). This is correct: the SVD/PCA components are learned globally, then applied locally. Verified by `test_subset_reduction` (fits on 200 rows, transforms 50 rows).

### Augmentation variance scaling — PASS
`sigma_i = U(sigma_range[0], sigma_range[1]) * feat_std` scales noise relative to feature magnitude. For zero-variance features, falls back to `feat_std = 1.0` (line 38). Verified by `test_zero_std_features`.

### Minor notes (non-blocking)
1. **BFS queue is a list with `pop(0)` — O(n) per dequeue.** For Cora subgraphs (n <= 200), this is negligible. A `collections.deque` would be O(1) but isn't necessary at this scale.
2. **Subgraph size can be < n_nodes** when `cc_size >= 0.8 * n_nodes`. The experiment script logs actual node count (`sample.graph.number_of_nodes()`) and stores it in results, so this is transparent. On Cora with n <= 200, this practically never triggers.

---

## 4. Test Results

### Phase 6 tests: 25/25 PASSED
```
tests/test_phase6.py  25 passed in 108.44s
```

Test coverage by module:
- **SubgraphSampler:** 8 tests (size, connectivity, relabeling, node_indices, edge preservation, reproducibility, different seeds, error on oversize)
- **MultipleSubgraphs:** 3 tests (count, type, diversity)
- **FeatureReducer:** 6 tests (SVD shape, PCA shape, variance preserved, subset reduction, reproducibility, invalid method error)
- **GaussianAugmentation:** 7 tests (shape, mean preservation, differ from original, reproducibility, different seeds, sigma range effect, zero-std handling)
- **FullPipeline:** 1 smoke test (Cora load -> BFS -> reduce -> augment -> train 5 epochs -> generate)

### Full test suite: 132/132 non-phase6 tests PASSED (no regressions)
```
tests/ (excluding test_phase6.py and test_downstream.py)  132 passed in 1209.42s
```

Phase 6 code introduces no regressions to existing phases.

---

## 5. Summary

| Check                         | Status |
|-------------------------------|--------|
| Data leakage: PCA fit scope   | PASS   |
| Data leakage: augmentation    | PASS   |
| Data leakage: importance wts  | N/A (6a baseline only) |
| Plan: BFS subgraph sampling   | PASS   |
| Plan: feature reduction       | PASS   |
| Plan: Gaussian augmentation   | PASS   |
| Plan: experiment grid         | PASS   |
| Plan: gate check              | PASS   |
| Correctness: node-feature map | PASS   |
| Correctness: reducer on subgraph | PASS |
| Correctness: augmentation stats | PASS |
| Tests: Phase 6 (25/25)       | PASS   |
| Tests: regression (132/132)   | PASS   |

No blocking issues found. All plan requirements met. All tests pass. No regressions.

---

**CLEAR: Proceed to runner.**
