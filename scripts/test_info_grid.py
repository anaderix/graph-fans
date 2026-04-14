"""Phase 4b: Test InfoGrid non-uniform DDIM step spacing.

Tests 3 axes:
- Grid type: uniform vs InfoGrid
- Step budget: 200, 100, 50 steps
- Training method: uniform vs band-shaped

Data leakage safeguards:
- Importance weights from seed-0 training split only.
- Reference split never accessed during training.

Usage:
    uv run python scripts/test_info_grid.py \\
        --families "SBM(q=0.05),BA(m=2)" \\
        --n-seeds 5 --device cuda \\
        --output results/phase4b/info_grid_results.json \\
        --profile-dir results/phase4a/entropy_rate_profiles \\
        --step-budgets "200,100,50"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from scipy.stats import ttest_rel

from graph_fans.phase0.spectral_profiler import (
    compute_laplacian_spectrum,
    partition_into_bands,
    compute_band_energy,
)
from graph_fans.phase2.dataset import get_or_generate_dataset
from graph_fans.phase2.evaluate import _get_graph
from graph_fans.phase2.info_grid import build_info_grid, build_uniform_grid
from graph_fans.phase2.noise_shaper import compute_importance_weights
from graph_fans.phase2.spectral_wasserstein import spectral_w1_summary
from graph_fans.phase2.trainer import Trainer, TrainConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

N_GEN = 50
DEFAULT_FAMILIES = "SBM(q=0.05),BA(m=2)"
DEFAULT_STEP_BUDGETS = "200,100,50"

# Training methods
TRAIN_METHODS = ["uniform", "band"]
# Grid types
GRID_TYPES = ["uniform_grid", "info_grid"]


def _load_profile(profile_dir: str, family: str, seed: int) -> dict | None:
    """Load entropy rate profile from saved JSON.

    Falls back to info_noise profile if info_noise+band is not available.
    """
    safe_family = family.replace("(", "_").replace(")", "").replace("=", "")
    # Try info_noise+band first, then info_noise
    for suffix in ["info_noise_band", "info_noise"]:
        path = Path(profile_dir) / f"{safe_family}_seed{seed}_{suffix}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


def _train_model(
    graph,
    train_data: np.ndarray,
    weights,
    train_method: str,
    seed: int,
    device: str,
    n_epochs: int,
    batch_timesteps: int,
    hidden_dim: int,
    n_layers: int,
) -> Trainer:
    """Train a model with the given method."""
    use_spectral = train_method == "band"

    cfg = TrainConfig(
        n_epochs=n_epochs,
        batch_timesteps=batch_timesteps,
        seed=seed,
        device=device,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        conv_type="gcn",
        sde_type="cosine",
        use_ema=True,
        use_lr_scheduler=True,
        n_train_samples=100,
        use_spectral_noise=use_spectral,
        t_sampling="uniform",
        record_entropy_rate=True,
        info_noise_n_bins=20,
    )
    trainer = Trainer(
        cfg, graph, train_data,
        importance_weights=weights if use_spectral else None,
    )
    trainer.train()
    return trainer


def run_family(
    family: str,
    n_nodes: int,
    n_seeds: int,
    step_budgets: list[int],
    device: str,
    dataset_dir: str,
    profile_dir: str | None,
    n_epochs: int = 500,
    batch_timesteps: int = 32,
    hidden_dim: int = 128,
    n_layers: int = 3,
) -> dict:
    """Run InfoGrid comparison for one graph family.

    Trains models with each training method, then generates with each
    grid type at each step budget.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"  {family} (n={n_nodes})")
    logger.info(f"{'='*80}")

    graph = _get_graph(family, n_nodes, seed=0)
    evals, evecs = compute_laplacian_spectrum(graph)
    _, bands = partition_into_bands(evals, B=8)

    # Compute importance weights from seed-0 TRAINING split only.
    ds0 = get_or_generate_dataset(
        graph, family, 0,
        n_train=100, n_ref=50,
        n_features=4, feature_mode="community",
        cache_dir=dataset_dir,
    )
    profiles_list = []
    for feat in ds0["train"][:50]:
        e = compute_band_energy(feat, evecs, bands)
        profiles_list.append(e)
    weights = compute_importance_weights(np.mean(profiles_list, axis=0))

    # Result tracking: key = (train_method, grid_type, n_steps)
    result_w1s: dict[tuple, list[float]] = {}
    per_seed: list[dict] = []

    for seed in range(n_seeds):
        ds = get_or_generate_dataset(
            graph, family, seed,
            n_train=100, n_ref=50,
            n_features=4, feature_mode="community",
            cache_dir=dataset_dir,
        )
        seed_entry: dict = {"seed": seed}

        for train_method in TRAIN_METHODS:
            logger.info(f"  Training {family}/seed={seed}/{train_method}...")
            trainer = _train_model(
                graph, ds["train"], weights, train_method, seed,
                device, n_epochs, batch_timesteps, hidden_dim, n_layers,
            )

            # Get entropy rate profile for InfoGrid
            profile = trainer.get_info_noise_profile()

            # Try loading a pre-computed profile if trainer profile is empty
            if profile is None and profile_dir is not None:
                profile = _load_profile(profile_dir, family, seed)

            for n_steps in step_budgets:
                for grid_type in GRID_TYPES:
                    key = (train_method, grid_type, n_steps)

                    if grid_type == "info_grid" and profile is not None:
                        ts = build_info_grid(profile, trainer.sde, n_steps=n_steps)
                        gen = trainer.generate_with_grid(ts, n_samples=N_GEN)
                    else:
                        # Uniform grid or no profile available
                        ts = build_uniform_grid(trainer.sde, n_steps=n_steps)
                        gen = trainer.generate_with_grid(ts, n_samples=N_GEN)

                    w1 = spectral_w1_summary(ds["ref"], gen, evecs, bands)

                    logger.info(
                        f"  {family}/seed={seed}/{train_method}/{grid_type}/{n_steps}steps: "
                        f"W1={w1['total_w1']:.1f}"
                    )

                    entry_key = f"{train_method}_{grid_type}_{n_steps}"
                    seed_entry[entry_key] = {
                        "w1_total": float(w1["total_w1"]),
                        "w1_low": float(w1["low_band_w1"]),
                        "w1_high": float(w1["high_band_w1"]),
                        "per_band_w1": w1["per_band_w1"].tolist(),
                    }

                    if key not in result_w1s:
                        result_w1s[key] = []
                    result_w1s[key].append(float(w1["total_w1"]))

        per_seed.append(seed_entry)

    # Statistical analysis
    # For each (train_method, n_steps): compare info_grid vs uniform_grid
    n_comparisons = len(TRAIN_METHODS) * len(step_budgets)
    bonferroni_alpha = 0.05 / max(n_comparisons, 1)

    comparisons: dict[str, dict] = {}
    for train_method in TRAIN_METHODS:
        for n_steps in step_budgets:
            key_uniform = (train_method, "uniform_grid", n_steps)
            key_info = (train_method, "info_grid", n_steps)

            if key_uniform not in result_w1s or key_info not in result_w1s:
                continue

            u_arr = np.array(result_w1s[key_uniform])
            i_arr = np.array(result_w1s[key_info])
            diff = u_arr - i_arr  # positive = info_grid is better

            if len(u_arr) >= 2:
                t_stat, p_val = ttest_rel(u_arr, i_arr)
            else:
                t_stat, p_val = 0.0, 1.0

            improvement_pct = float(diff.mean() / u_arr.mean() * 100) if u_arr.mean() > 0 else 0.0
            comp_key = f"{train_method}_{n_steps}steps"

            comparisons[comp_key] = {
                "train_method": train_method,
                "n_steps": n_steps,
                "uniform_w1_mean": float(u_arr.mean()),
                "uniform_w1_std": float(u_arr.std()),
                "info_w1_mean": float(i_arr.mean()),
                "info_w1_std": float(i_arr.std()),
                "improvement_pct": improvement_pct,
                "t_stat": float(t_stat),
                "p_val": float(p_val),
                "significant": bool(p_val < bonferroni_alpha),
            }

    # Summary log
    logger.info(f"\n  {family} SUMMARY:")
    for comp_key, comp in comparisons.items():
        sig = "SIGNIFICANT" if comp["significant"] else "not significant"
        logger.info(
            f"    {comp_key}: uniform={comp['uniform_w1_mean']:.1f}, "
            f"info={comp['info_w1_mean']:.1f}, "
            f"improvement={comp['improvement_pct']:.1f}%, "
            f"p={comp['p_val']:.4f} {sig}"
        )

    return {
        "family": family,
        "n_nodes": n_nodes,
        "step_budgets": step_budgets,
        "results": {
            f"{k[0]}_{k[1]}_{k[2]}": {
                "w1s": v,
                "w1_mean": float(np.mean(v)),
                "w1_std": float(np.std(v)),
            }
            for k, v in result_w1s.items()
        },
        "comparisons": comparisons,
        "per_seed": per_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4b: Test InfoGrid non-uniform DDIM step spacing"
    )
    parser.add_argument(
        "--families", type=str, default=DEFAULT_FAMILIES,
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
        "--step-budgets", type=str, default=DEFAULT_STEP_BUDGETS,
        help=f"Comma-separated step budgets (default: {DEFAULT_STEP_BUDGETS})",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device: 'cuda' or 'cpu' (default: cuda)",
    )
    parser.add_argument(
        "--output", type=str, default="results/phase4b/info_grid_results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--dataset-dir", type=str, default="results/phase4a/datasets",
        help="Dataset cache directory",
    )
    parser.add_argument(
        "--profile-dir", type=str, default="results/phase4a/entropy_rate_profiles",
        help="Directory with saved entropy rate profiles",
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

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    step_budgets = [int(s.strip()) for s in args.step_budgets.split(",") if s.strip()]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    for family in families:
        result = run_family(
            family=family,
            n_nodes=args.n_nodes,
            n_seeds=args.n_seeds,
            step_budgets=step_budgets,
            device=args.device,
            dataset_dir=args.dataset_dir,
            profile_dir=args.profile_dir,
            n_epochs=args.n_epochs,
            batch_timesteps=args.batch_timesteps,
            hidden_dim=args.hidden_dim,
        )
        all_results.append(result)

        # Write after each family (crash-safe)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"  Saved intermediate results to {output_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print("W1 by (training method, grid type, step budget):")
    print("-" * 100)
    print(f"{'Config':<40} {'W1 mean':>10} {'W1 std':>10}")
    print("-" * 100)
    for r in all_results:
        print(f"\n  {r['family']}:")
        for key, val in sorted(r["results"].items()):
            print(f"    {key:<36} {val['w1_mean']:>10.1f} {val['w1_std']:>10.1f}")

    print("\n" + "-" * 100)
    print("InfoGrid vs Uniform Grid comparisons:")
    print(f"{'Config':<30} {'Improv%':>10} {'p-val':>10} {'Sig?':>6}")
    print("-" * 100)
    for r in all_results:
        for comp_key, comp in r["comparisons"].items():
            sig_mark = "*" if comp["significant"] else " "
            print(
                f"  {r['family']}/{comp_key:<22} "
                f"{comp['improvement_pct']:>9.1f}% "
                f"{comp['p_val']:>10.4f} "
                f"{sig_mark:>6}"
            )
    print("=" * 100)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
