"""Phase 4a: Test InfoNoise adaptive timestep sampling vs baselines.

4-way comparison:
1. uniform    -- baseline uniform t-sampling
2. band       -- Phase 2g band-mismatched spectral shaping
3. info_noise -- InfoNoise adaptive t-sampling, no spectral shaping
4. info_noise+band -- InfoNoise + band-mismatched shaping

Data leakage safeguards:
- Importance weights are computed from seed-0 training split only.
  ds["ref"] is never touched during weight computation.
- Separate --dataset-dir per scale to prevent cache key collisions.
- Each seed uses an independently generated dataset split.

Usage:
    uv run python scripts/test_info_noise.py \\
        --families "SBM(q=0.05),BA(m=2)" \\
        --n-nodes 50 --n-seeds 5 --device cuda \\
        --output results/phase4a/info_noise_results.json \\
        --dataset-dir results/phase4a/datasets \\
        --n-epochs 500 --save-profiles
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
from graph_fans.phase2.noise_shaper import compute_importance_weights
from graph_fans.phase2.spectral_wasserstein import spectral_w1_summary
from graph_fans.phase2.trainer import Trainer, TrainConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

N_GEN = 50
DEFAULT_FAMILIES = "SBM(q=0.05),BA(m=2)"

# 4 methods to compare
METHODS = ["uniform", "band", "info_noise", "info_noise+band"]


def _make_config(
    method: str,
    seed: int,
    device: str,
    n_epochs: int,
    batch_timesteps: int,
    hidden_dim: int,
    n_layers: int,
) -> TrainConfig:
    """Create TrainConfig for a given method."""
    use_spectral = method in ("band", "info_noise+band")
    use_info_noise = method in ("info_noise", "info_noise+band")

    return TrainConfig(
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
        t_sampling="info_noise" if use_info_noise else "uniform",
        info_noise_warm_up_steps=2000,
        info_noise_n_bins=20,
    )


def run_family(
    family: str,
    n_nodes: int,
    n_seeds: int,
    device: str,
    dataset_dir: str,
    n_epochs: int = 500,
    batch_timesteps: int = 32,
    hidden_dim: int = 128,
    n_layers: int = 3,
    save_profiles: bool = False,
    profile_dir: str | None = None,
) -> dict:
    """Run 4-way comparison for one graph family.

    Importance weights are derived from seed-0 training split only.
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
    profiles = []
    for feat in ds0["train"][:50]:
        e = compute_band_energy(feat, evecs, bands)
        profiles.append(e)
    weights = compute_importance_weights(np.mean(profiles, axis=0))
    logger.info(f"  Importance weights (from seed-0 train): {weights.weights.round(3)}")

    # Per-method W1 tracking
    method_w1s: dict[str, list[float]] = {m: [] for m in METHODS}
    per_seed: list[dict] = []

    for seed in range(n_seeds):
        ds = get_or_generate_dataset(
            graph, family, seed,
            n_train=100, n_ref=50,
            n_features=4, feature_mode="community",
            cache_dir=dataset_dir,
        )

        seed_entry: dict = {"seed": seed}

        for method in METHODS:
            cfg = _make_config(
                method, seed, device, n_epochs,
                batch_timesteps, hidden_dim, n_layers,
            )

            use_spectral = method in ("band", "info_noise+band")
            trainer = Trainer(
                cfg, graph, ds["train"],
                importance_weights=weights if use_spectral else None,
            )
            history = trainer.train()

            # Save entropy rate profile if requested
            if save_profiles and profile_dir and "info_noise" in method:
                profile = trainer.get_info_noise_profile()
                if profile is not None:
                    pdir = Path(profile_dir)
                    pdir.mkdir(parents=True, exist_ok=True)
                    safe_family = family.replace("(", "_").replace(")", "").replace("=", "")
                    ppath = pdir / f"{safe_family}_seed{seed}_{method.replace('+', '_')}.json"
                    with open(ppath, "w") as f:
                        json.dump(profile, f, indent=2)
                    logger.info(f"  Saved entropy rate profile to {ppath}")

            gen = trainer.generate(n_samples=N_GEN)
            w1 = spectral_w1_summary(ds["ref"], gen, evecs, bands)
            sanity = trainer.sanity_check(ds["train"], n_gen=5)

            logger.info(
                f"  {family}/seed={seed}/{method}: "
                f"loss={history['loss'][-1]:.3f}, "
                f"std_ratio={sanity['std_ratio']:.2f}, "
                f"W1={w1['total_w1']:.1f} "
                f"(low={w1['low_band_w1']:.1f}, high={w1['high_band_w1']:.1f})"
            )

            seed_entry[method] = {
                "final_loss": float(history["loss"][-1]),
                "std_ratio": float(sanity["std_ratio"]),
                "w1_total": float(w1["total_w1"]),
                "w1_low": float(w1["low_band_w1"]),
                "w1_high": float(w1["high_band_w1"]),
                "per_band_w1": w1["per_band_w1"].tolist(),
            }
            method_w1s[method].append(float(w1["total_w1"]))

        per_seed.append(seed_entry)

    # Statistical analysis: paired t-tests with Bonferroni correction
    # Compare each method against uniform baseline
    n_comparisons = len(METHODS) - 1  # all vs uniform
    bonferroni_alpha = 0.05 / n_comparisons

    u_arr = np.array(method_w1s["uniform"])
    comparisons: dict[str, dict] = {}

    for method in METHODS:
        if method == "uniform":
            continue
        m_arr = np.array(method_w1s[method])
        diff = u_arr - m_arr  # positive = method is better

        if len(u_arr) >= 2:
            t_stat, p_val = ttest_rel(u_arr, m_arr)
        else:
            t_stat, p_val = 0.0, 1.0

        improvement_pct = float(diff.mean() / u_arr.mean() * 100) if u_arr.mean() > 0 else 0.0
        significant = bool(p_val < bonferroni_alpha)

        comparisons[method] = {
            "improvement_pct": improvement_pct,
            "t_stat": float(t_stat),
            "p_val": float(p_val),
            "significant": significant,
        }

    # Summary
    logger.info(f"\n  {family} SUMMARY:")
    for method in METHODS:
        arr = np.array(method_w1s[method])
        logger.info(f"    {method:20s}: W1={arr.mean():.1f} +/- {arr.std():.1f}")
    for method, comp in comparisons.items():
        sig = "SIGNIFICANT" if comp["significant"] else "not significant"
        logger.info(
            f"    {method} vs uniform: {comp['improvement_pct']:.1f}% "
            f"(t={comp['t_stat']:.3f}, p={comp['p_val']:.4f}) {sig}"
        )

    return {
        "family": family,
        "n_nodes": n_nodes,
        "methods": {
            m: {
                "w1s": method_w1s[m],
                "w1_mean": float(np.array(method_w1s[m]).mean()),
                "w1_std": float(np.array(method_w1s[m]).std()),
            }
            for m in METHODS
        },
        "comparisons": comparisons,
        "per_seed": per_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4a: Test InfoNoise adaptive timestep sampling"
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
        "--device", type=str, default="cuda",
        help="Torch device: 'cuda' or 'cpu' (default: cuda)",
    )
    parser.add_argument(
        "--output", type=str, default="results/phase4a/info_noise_results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--dataset-dir", type=str, default="results/phase4a/datasets",
        help="Dataset cache directory (use separate dir per n_nodes)",
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
    parser.add_argument(
        "--save-profiles", action="store_true",
        help="Save entropy rate profiles as JSON",
    )
    parser.add_argument(
        "--profile-dir", type=str, default="results/phase4a/entropy_rate_profiles",
        help="Directory to save entropy rate profiles",
    )
    args = parser.parse_args()

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    for family in families:
        result = run_family(
            family=family,
            n_nodes=args.n_nodes,
            n_seeds=args.n_seeds,
            device=args.device,
            dataset_dir=args.dataset_dir,
            n_epochs=args.n_epochs,
            batch_timesteps=args.batch_timesteps,
            hidden_dim=args.hidden_dim,
            save_profiles=args.save_profiles,
            profile_dir=args.profile_dir,
        )
        all_results.append(result)

        # Write after each family (crash-safe)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"  Saved intermediate results to {output_path}")

    # Print summary table
    print("\n" + "=" * 100)
    header = f"{'Family':<20}"
    for m in METHODS:
        header += f" {m:>15}"
    print(header)
    print("-" * 100)
    for r in all_results:
        row = f"{r['family']:<20}"
        for m in METHODS:
            w1 = r["methods"][m]["w1_mean"]
            row += f" {w1:>15.1f}"
        print(row)
    print("-" * 100)

    # Comparisons
    print(f"\n{'Method vs uniform':<25} {'Improv%':>10} {'p-val':>10} {'Sig?':>6}")
    print("-" * 60)
    for r in all_results:
        for m, comp in r["comparisons"].items():
            sig_mark = "*" if comp["significant"] else " "
            print(
                f"  {r['family']}/{m:<15} "
                f"{comp['improvement_pct']:>9.1f}% "
                f"{comp['p_val']:>10.4f} "
                f"{sig_mark:>6}"
            )
    print("=" * 100)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
