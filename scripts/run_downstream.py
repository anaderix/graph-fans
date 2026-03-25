"""CLI entry point for downstream node classification evaluation.

Evaluates whether W1 improvements from spectral noise shaping translate to
practical gains on node classification using generated features.

Usage:
    uv run python scripts/run_downstream.py \
        --families SBM(q=0.05),BA(m=5) \
        --n-nodes 50 \
        --n-seeds 5 \
        --device cuda \
        --n-gen-samples 1000 \
        --output results/diagnostics/downstream_results.json \
        --dataset-dir results/phase2f_small/datasets
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_FAMILIES = "SBM(q=0.05),BA(m=5)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run downstream node classification evaluation for Graph-FANS"
    )
    parser.add_argument(
        "--families",
        type=str,
        default=DEFAULT_FAMILIES,
        help=f"Comma-separated graph families (default: {DEFAULT_FAMILIES})",
    )
    parser.add_argument(
        "--n-nodes", type=int, default=50,
        help="Number of nodes per graph (default: 50)",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=5,
        help="Number of random seeds per family (default: 5)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device: 'cuda' or 'cpu' (default: cuda)",
    )
    parser.add_argument(
        "--n-gen-samples", type=int, default=1000,
        help="Number of generated samples for classifier training (default: 1000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/diagnostics/downstream_results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="results/phase2f_small/datasets",
        help="Dataset cache directory",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=500,
        help="Training epochs per run (default: 500)",
    )
    parser.add_argument(
        "--batch-timesteps", type=int, default=32,
        help="Batch timesteps per epoch (default: 32)",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=128,
        help="Score network hidden dimension (default: 128)",
    )
    args = parser.parse_args()

    from graph_fans.phase2.downstream import run_downstream_experiment

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    for family in families:
        result = run_downstream_experiment(
            family=family,
            n_nodes=args.n_nodes,
            n_seeds=args.n_seeds,
            device=args.device,
            dataset_dir=args.dataset_dir,
            n_gen_samples=args.n_gen_samples,
            n_epochs=args.n_epochs,
            batch_timesteps=args.batch_timesteps,
            hidden_dim=args.hidden_dim,
        )
        all_results.append(result)

        # Write after each family (crash-safe)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"  Saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Family':<20} {'Unif Acc':>10} {'Spec Acc':>10} {'Improv%':>10} {'p-val':>10} {'Sig?':>6}")
    print("-" * 80)
    for r in all_results:
        sig = "*" if r["significant"] else " "
        print(
            f"{r['family']:<20} "
            f"{r['uniform_acc_mean']:>10.3f} "
            f"{r['spectral_acc_mean']:>10.3f} "
            f"{r['improvement_pct']:>9.1f}% "
            f"{r['p_val']:>10.4f} "
            f"{sig:>6}"
        )
    print("=" * 80)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
