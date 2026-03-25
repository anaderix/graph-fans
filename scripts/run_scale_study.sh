#!/usr/bin/env bash
# Run scale study: SBM(q=0.05) and BA(m=5) at n_nodes in {100, 150, 200}.
#
# Each scale uses a SEPARATE dataset directory to prevent cache key collisions.
# Cache key is {family}_seed{seed}.npz and does NOT encode n_nodes.
#
# Usage:
#   cd /home/anaderi/projects/graph-fans
#   bash scripts/run_scale_study.sh [--device cuda] [--n-seeds 5]
#
# Outputs:
#   results/diagnostics/scale_study_n100.json
#   results/diagnostics/scale_study_n150.json
#   results/diagnostics/scale_study_n200.json
#   results/diagnostics/scale_study_n200_hd256.json  (if sanity check triggers retry)

set -euo pipefail

DEVICE="${DEVICE:-cuda}"
N_SEEDS="${N_SEEDS:-5}"
FAMILIES="SBM(q=0.05),BA(m=5)"

echo "=== Graph-FANS Scale Study ==="
echo "Device: ${DEVICE}, N_seeds: ${N_SEEDS}"
echo ""

# SCALE-1: n=100
echo "--- SCALE-1: n=100 ---"
uv run python scripts/test_shaping_w1.py \
    --families "${FAMILIES}" \
    --n-nodes 100 \
    --n-seeds "${N_SEEDS}" \
    --device "${DEVICE}" \
    --output results/diagnostics/scale_study_n100.json \
    --dataset-dir results/phase2_scale/datasets_n100

echo ""

# SCALE-2: n=150
echo "--- SCALE-2: n=150 ---"
uv run python scripts/test_shaping_w1.py \
    --families "${FAMILIES}" \
    --n-nodes 150 \
    --n-seeds "${N_SEEDS}" \
    --device "${DEVICE}" \
    --output results/diagnostics/scale_study_n150.json \
    --dataset-dir results/phase2_scale/datasets_n150

echo ""

# SCALE-3: n=200
echo "--- SCALE-3: n=200 ---"
uv run python scripts/test_shaping_w1.py \
    --families "${FAMILIES}" \
    --n-nodes 200 \
    --n-seeds "${N_SEEDS}" \
    --device "${DEVICE}" \
    --output results/diagnostics/scale_study_n200.json \
    --dataset-dir results/phase2_scale/datasets_n200

echo ""

# Check if n=200 results show high std_ratio (model capacity limit)
# If so, trigger SCALE-4 retry with hidden_dim=256.
STD_RATIO_CHECK=$(uv run python - << 'PYEOF'
import json, sys
try:
    with open("results/diagnostics/scale_study_n200.json") as f:
        data = json.load(f)
    high_ratio_count = 0
    for family_data in data:
        for seed_data in family_data.get("per_seed", []):
            for method in ["uniform", "spectral"]:
                sr = seed_data.get(method, {}).get("std_ratio", 0)
                if sr > 3:
                    high_ratio_count += 1
    print(high_ratio_count)
except Exception as e:
    print(0)
PYEOF
)

if [ "${STD_RATIO_CHECK}" -gt 3 ]; then
    echo "--- SCALE-4: n=200 retry with hidden_dim=256 (std_ratio > 3 detected in ${STD_RATIO_CHECK} seeds) ---"
    uv run python scripts/test_shaping_w1.py \
        --families "${FAMILIES}" \
        --n-nodes 200 \
        --n-seeds "${N_SEEDS}" \
        --device "${DEVICE}" \
        --hidden-dim 256 \
        --output results/diagnostics/scale_study_n200_hd256.json \
        --dataset-dir results/phase2_scale/datasets_n200
    echo ""
else
    echo "--- SCALE-4 not needed: std_ratio within bounds at n=200 ---"
fi

echo "=== Scale study complete ==="
echo "Run 'uv run python scripts/analyze_scale_results.py' to generate plots."
