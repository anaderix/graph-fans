"""Phase 3b: Test matched generation noise vs mismatched vs uniform.

3-way comparison using band shaping only:
  1. uniform     — uniform training noise, uniform generation start
  2. band-mismatched — band-shaped training noise, uniform generation start
  3. band-matched   — band-shaped training noise, band-shaped generation start

Data leakage safeguards:
- Band weights are computed from seed-0 training split only.
  ds["ref"] is never touched during weight computation.
- Separate --dataset-dir per scale to prevent cache key collisions.
- Each seed uses an independently generated dataset split.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

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
DEFAULT_FAMILIES = "SBM(q=0.05),SBM(q=0.1),BA(m=2),SBM(q=0.01)"

# Condition definitions: (label, use_spectral_noise, shape_gen_noise)
CONDITIONS = [
    ("uniform",         False, False),
    ("band-mismatched", True,  False),
    ("band-matched",    True,  True),
]


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
) -> dict:
    """Run 3-condition comparison for one graph family.

    Band importance weights are derived from seed-0 training split only.
    ds["ref"] is never accessed during weight computation.

    Args:
        family: Graph family string, e.g. "SBM(q=0.1)" or "BA(m=2)".
        n_nodes: Number of graph nodes.
        n_seeds: Number of seeds to run.
        device: Torch device string ("cpu" or "cuda").
        dataset_dir: Root directory for dataset cache.
        n_epochs: Training epochs per run.
        batch_timesteps: Batch timesteps per epoch.
        hidden_dim: Score network hidden dimension.
        n_layers: Score network GCN layers.

    Returns:
        Dict with family, n_nodes, per-condition W1 lists, pairwise stats, per_seed.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"  {family} (n={n_nodes})")
    logger.info(f"{'='*80}")

    graph = _get_graph(family, n_nodes, seed=0)
    evals, evecs = compute_laplacian_spectrum(graph)
    _, bands = partition_into_bands(evals, B=8)

    # Compute band weights from seed-0 TRAINING split only.
    ds0 = get_or_generate_dataset(
        graph, family, 0,
        n_train=100, n_ref=50,
        n_features=4, feature_mode="community",
        cache_dir=dataset_dir,
    )

    band_profiles = []
    for feat in ds0["train"][:50]:
        e = compute_band_energy(feat, evecs, bands)
        band_profiles.append(e)
    band_weights = compute_importance_weights(np.mean(band_profiles, axis=0))
    logger.info(f"  Band weights (from seed-0 train): {band_weights.weights.round(3)}")

    condition_labels = [c[0] for c in CONDITIONS]
    w1s: dict[str, list[float]] = {label: [] for label in condition_labels}
    per_seed: list[dict] = []

    for seed in range(n_seeds):
        ds = get_or_generate_dataset(
            graph, family, seed,
            n_train=100, n_ref=50,
            n_features=4, feature_mode="community",
            cache_dir=dataset_dir,
        )

        seed_entry: dict = {"seed": seed}

        for label, use_spectral, shape_gen in CONDITIONS:
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
                shape_gen_noise=shape_gen,
            )

            iw = band_weights if use_spectral else None

            trainer = Trainer(
                cfg, graph, ds["train"],
                importance_weights=iw,
            )
            history = trainer.train()

            gen = trainer.generate(n_samples=N_GEN)
            w1 = spectral_w1_summary(ds["ref"], gen, evecs, bands)
            sanity = trainer.sanity_check(ds["train"], n_gen=5)

            logger.info(
                f"  {family}/seed={seed}/{label}: "
                f"loss={history['loss'][-1]:.3f}, "
                f"std_ratio={sanity['std_ratio']:.2f}, "
                f"W1={w1['total_w1']:.1f} "
                f"(low={w1['low_band_w1']:.1f}, high={w1['high_band_w1']:.1f})"
            )

            seed_entry[label] = {
                "final_loss": float(history["loss"][-1]),
                "std_ratio": float(sanity["std_ratio"]),
                "w1_total": float(w1["total_w1"]),
                "w1_low": float(w1["low_band_w1"]),
                "w1_high": float(w1["high_band_w1"]),
                "per_band_w1": w1["per_band_w1"].tolist(),
            }

            w1s[label].append(float(w1["total_w1"]))

        per_seed.append(seed_entry)

    # Pairwise comparisons
    comparisons = [
        ("uniform", "band-mismatched"),
        ("uniform", "band-matched"),
        ("band-mismatched", "band-matched"),  # key comparison
    ]
    n_families_for_bonferroni = 4
    adjusted_alpha = 0.05 / n_families_for_bonferroni

    pairwise: dict[str, dict] = {}
    for m_a, m_b in comparisons:
        a = np.array(w1s[m_a])
        b = np.array(w1s[m_b])
        diff = a - b

        if len(a) >= 2:
            t_stat, p_val = ttest_rel(a, b)
        else:
            t_stat, p_val = 0.0, 1.0

        improvement_pct = float(diff.mean() / a.mean() * 100) if a.mean() > 0 else 0.0
        significant = bool(p_val < adjusted_alpha)

        key = f"{m_a}_vs_{m_b}"
        pairwise[key] = {
            "a_condition": m_a,
            "b_condition": m_b,
            "a_w1_mean": float(a.mean()),
            "a_w1_std": float(a.std()),
            "b_w1_mean": float(b.mean()),
            "b_w1_std": float(b.std()),
            "improvement_pct": improvement_pct,
            "t_stat": float(t_stat),
            "p_val": float(p_val),
            "significant": significant,
            "adjusted_alpha": adjusted_alpha,
        }

        logger.info(f"\n  {family} {m_a} vs {m_b}:")
        logger.info(f"    {m_a} W1: {a.mean():.1f} +/- {a.std():.1f}")
        logger.info(f"    {m_b} W1: {b.mean():.1f} +/- {b.std():.1f}")
        logger.info(f"    Improvement: {diff.mean():.1f} ({improvement_pct:.1f}%)")
        logger.info(f"    t={t_stat:.3f}, p={p_val:.4f}")
        logger.info(f"    {'SIGNIFICANT' if significant else 'not significant'} (Bonferroni alpha={adjusted_alpha:.4f})")

    return {
        "family": family,
        "n_nodes": n_nodes,
        "uniform_w1s": w1s["uniform"],
        "band_mismatched_w1s": w1s["band-mismatched"],
        "band_matched_w1s": w1s["band-matched"],
        "pairwise": pairwise,
        "per_seed": per_seed,
        "band_weights": band_weights.weights.tolist(),
    }


def validate_only(
    families: list[str],
    n_nodes: int,
    dataset_dir: str,
) -> None:
    """Print weight profiles without training -- for quick sanity checks."""
    for family in families:
        graph = _get_graph(family, n_nodes, seed=0)
        evals, evecs = compute_laplacian_spectrum(graph)
        _, bands = partition_into_bands(evals, B=8)

        ds0 = get_or_generate_dataset(
            graph, family, 0,
            n_train=100, n_ref=50,
            n_features=4, feature_mode="community",
            cache_dir=dataset_dir,
        )

        band_profiles = []
        for feat in ds0["train"][:50]:
            e = compute_band_energy(feat, evecs, bands)
            band_profiles.append(e)
        bw = compute_importance_weights(np.mean(band_profiles, axis=0))

        print(f"\n{'='*60}")
        print(f"{family} (n={n_nodes})")
        print(f"{'='*60}")
        print(f"  Band energies: {bw.band_energies.round(2)}")
        print(f"  Band weights:  {bw.weights.round(3)}")
        print(f"  Conditions to test:")
        for label, use_spectral, shape_gen in CONDITIONS:
            print(f"    {label}: use_spectral_noise={use_spectral}, shape_gen_noise={shape_gen}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3b: 3-way comparison of uniform vs band-mismatched vs band-matched noise"
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
        "--output",
        type=str,
        default="results/phase3b/phase3b_results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="results/phase3b/datasets",
        help=(
            "Dataset cache directory. Use a SEPARATE directory per --n-nodes value "
            "to prevent stale cache collisions (cache key does not encode n_nodes)."
        ),
    )
    parser.add_argument(
        "--n-epochs", type=int, default=500,
        help="Training epochs per run (default: 500)",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Print weight profiles without training",
    )
    args = parser.parse_args()

    families = [f.strip() for f in args.families.split(",") if f.strip()]

    if args.validate_only:
        validate_only(families, args.n_nodes, args.dataset_dir)
        return

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
        )
        all_results.append(result)

        # Write after each family (crash-safe)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"  Saved intermediate results to {output_path}")

    # Print summary table
    condition_labels = [c[0] for c in CONDITIONS]
    print(f"\n{'='*110}")
    print(f"{'Family':<20} {'Uniform W1':>12} {'Mismatch W1':>12} {'Matched W1':>12} "
          f"{'U-Mis Imp%':>12} {'U-Mat Imp%':>12} {'Mis-Mat Imp%':>14}")
    print(f"{'-'*110}")
    for r in all_results:
        pw = r["pairwise"]
        u_mean = np.mean(r["uniform_w1s"])
        mis_mean = np.mean(r["band_mismatched_w1s"])
        mat_mean = np.mean(r["band_matched_w1s"])

        u_mis = pw["uniform_vs_band-mismatched"]
        u_mat = pw["uniform_vs_band-matched"]
        mis_mat = pw["band-mismatched_vs_band-matched"]

        u_mis_sig = "*" if u_mis["significant"] else " "
        u_mat_sig = "*" if u_mat["significant"] else " "
        mis_mat_sig = "*" if mis_mat["significant"] else " "

        print(
            f"{r['family']:<20} "
            f"{u_mean:>12.1f} "
            f"{mis_mean:>12.1f} "
            f"{mat_mean:>12.1f} "
            f"{u_mis['improvement_pct']:>10.1f}%{u_mis_sig} "
            f"{u_mat['improvement_pct']:>10.1f}%{u_mat_sig} "
            f"{mis_mat['improvement_pct']:>12.1f}%{mis_mat_sig} "
        )
    print(f"{'='*110}")
    print("* = significant at Bonferroni-corrected alpha=0.0125")
    print("Key comparison: band-mismatched vs band-matched (last column)")

    # Detailed pairwise table
    print(f"\n{'='*110}")
    print(f"{'Family':<20} {'Comparison':<40} {'p-val':>10} {'t-stat':>10} {'Imp%':>10} {'Sig?':>6}")
    print(f"{'-'*110}")
    for r in all_results:
        for key, pw in r["pairwise"].items():
            sig_mark = "*" if pw["significant"] else " "
            print(
                f"{r['family']:<20} "
                f"{key:<40} "
                f"{pw['p_val']:>10.4f} "
                f"{pw['t_stat']:>10.3f} "
                f"{pw['improvement_pct']:>9.1f}% "
                f"{sig_mark:>6}"
            )
    print(f"{'='*110}")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
