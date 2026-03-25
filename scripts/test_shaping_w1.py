"""Test whether spectral noise shaping improves W1 on generated samples.

Trains a 3-layer GCN with uniform vs spectral noise on configurable graph
families, generates samples, compares per-band W1. Repeats over multiple seeds
for statistical robustness.

Data leakage safeguards:
- Importance weights are computed from ds["train"] (the training split).
  ds["ref"] is never touched during weight computation.
- Separate --dataset-dir per scale to prevent cache key collisions
  (cache key is {family}_seed{seed}.npz, does NOT encode n_nodes).
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
DEFAULT_FAMILIES = "SBM(q=0.1),BA(m=2),BA(m=5)"


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
    """Run uniform vs spectral training comparison for one graph family.

    Importance weights are derived from seed-0 training split only.
    ds["ref"] is never accessed during weight computation.

    Args:
        family: Graph family string, e.g. "SBM(q=0.1)" or "BA(m=2)".
        n_nodes: Number of graph nodes.
        n_seeds: Number of seeds to run.
        device: Torch device string ("cpu" or "cuda").
        dataset_dir: Root directory for dataset cache.
            MUST be separate per n_nodes to prevent stale cache loading.
        n_epochs: Training epochs per run.
        batch_timesteps: Batch timesteps per epoch.
        hidden_dim: Score network hidden dimension.
        n_layers: Score network GCN layers.

    Returns:
        Dict with keys: family, n_nodes, uniform_w1s, spectral_w1s,
        t_stat, p_val, improvement_pct, significant, per_seed.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"  {family} (n={n_nodes})")
    logger.info(f"{'='*80}")

    graph = _get_graph(family, n_nodes, seed=0)
    evals, evecs = compute_laplacian_spectrum(graph)
    _, bands = partition_into_bands(evals, B=8)

    # Compute importance weights from seed-0 TRAINING split only.
    # ds0["ref"] is never accessed here.
    ds0 = get_or_generate_dataset(
        graph, family, 0,
        n_train=100, n_ref=50,
        n_features=4, feature_mode="community",
        cache_dir=dataset_dir,
    )
    profiles = []
    for feat in ds0["train"][:50]:  # first 50 training samples for weight estimation
        e = compute_band_energy(feat, evecs, bands)
        profiles.append(e)
    weights = compute_importance_weights(np.mean(profiles, axis=0))
    logger.info(f"  Importance weights (from seed-0 train): {weights.weights.round(3)}")

    uniform_w1s: list[float] = []
    spectral_w1s: list[float] = []
    per_seed: list[dict] = []

    for seed in range(n_seeds):
        ds = get_or_generate_dataset(
            graph, family, seed,
            n_train=100, n_ref=50,
            n_features=4, feature_mode="community",
            cache_dir=dataset_dir,
        )

        seed_entry: dict = {"seed": seed}

        for method, use_spectral in [("uniform", False), ("spectral", True)]:
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
            )
            trainer = Trainer(
                cfg, graph, ds["train"],
                importance_weights=weights if use_spectral else None,
            )
            history = trainer.train()

            # Generate samples — torch.randn inside generate() is called per sample
            # with no manual seed, so each of N_GEN samples is independent.
            gen = trainer.generate(n_samples=N_GEN)

            # Evaluate W1 against the reference split (never used during training).
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

            if method == "uniform":
                uniform_w1s.append(float(w1["total_w1"]))
            else:
                spectral_w1s.append(float(w1["total_w1"]))

        per_seed.append(seed_entry)

    # Paired t-test
    u = np.array(uniform_w1s)
    s = np.array(spectral_w1s)
    diff = u - s

    if len(u) >= 2:
        t_stat, p_val = ttest_rel(u, s)
    else:
        t_stat, p_val = 0.0, 1.0

    improvement_pct = float(diff.mean() / u.mean() * 100) if u.mean() > 0 else 0.0
    significant = bool(p_val < 0.025)  # Bonferroni-adjusted alpha

    logger.info(f"\n  {family} SUMMARY:")
    logger.info(f"    Uniform W1:  {u.mean():.1f} +/- {u.std():.1f}")
    logger.info(f"    Spectral W1: {s.mean():.1f} +/- {s.std():.1f}")
    logger.info(f"    Improvement: {diff.mean():.1f} ({improvement_pct:.1f}%)")
    logger.info(f"    t={t_stat:.3f}, p={p_val:.4f}")
    logger.info(f"    {'SIGNIFICANT' if significant else 'not significant'} (Bonferroni alpha=0.025)")

    return {
        "family": family,
        "n_nodes": n_nodes,
        "uniform_w1s": uniform_w1s,
        "spectral_w1s": spectral_w1s,
        "uniform_w1_mean": float(u.mean()),
        "uniform_w1_std": float(u.std()),
        "spectral_w1_mean": float(s.mean()),
        "spectral_w1_std": float(s.std()),
        "improvement_pct": improvement_pct,
        "t_stat": float(t_stat),
        "p_val": float(p_val),
        "significant": significant,
        "per_seed": per_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test spectral noise shaping W1 improvement across graph families"
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
        default="results/diagnostics/family_generalization.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="results/phase2f_small/datasets",
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
        "--batch-timesteps", type=int, default=32,
        help="Batch timesteps per epoch (default: 32)",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=128,
        help="Score network hidden dimension (default: 128)",
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
        )
        all_results.append(result)

        # Write after each family (crash-safe)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"  Saved intermediate results to {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Family':<20} {'Unif W1':>10} {'Spec W1':>10} {'Improv%':>10} {'p-val':>10} {'Sig?':>6}")
    print("-" * 80)
    n_significant = 0
    for r in all_results:
        sig_mark = "*" if r["significant"] else " "
        print(
            f"{r['family']:<20} "
            f"{r['uniform_w1_mean']:>10.1f} "
            f"{r['spectral_w1_mean']:>10.1f} "
            f"{r['improvement_pct']:>9.1f}% "
            f"{r['p_val']:>10.4f} "
            f"{sig_mark:>6}"
        )
        if r["significant"]:
            n_significant += 1
    print("=" * 80)
    print(f"Families with significant improvement (p<0.025): {n_significant}/{len(all_results)}")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
