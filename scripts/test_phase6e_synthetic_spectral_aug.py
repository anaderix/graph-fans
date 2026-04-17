"""Phase 6e: Test band shaping on synthetic SBM features with spectral augmentation.

Diagnostic to disambiguate why Phase 6c failed:
- H1: Spectral augmentation makes band shaping redundant (augmentation already shapes spectrally)
- H2: Cora's PCA-reduced features lack bimodal spectral structure
- H3: N_ref=1 evaluation noise masks a small effect

This experiment isolates H1. Uses SBM(q=0.05) community-boundary features
(where Phase 2g showed 12.5% W1 improvement with independent samples), but
replaces the 100 independent training samples with 100 spectral-augmented
copies from a single seed.

If band shaping still helps → H1 rejected, 6c failure is feature-specific.
If band shaping doesn't help → H1 confirmed, spectral augmentation
eliminates the shaping effect regardless of features.

Matches Phase 2g config: SBM(q=0.05), n=50, d=4, 500 epochs, 3L GCN 128h, 5 seeds.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

from graph_fans.phase0.spectral_profiler import (
    compute_band_energy,
    compute_laplacian_spectrum,
    partition_into_bands,
)
from graph_fans.phase2.noise_shaper import compute_importance_weights
from graph_fans.phase2.spectral_wasserstein import spectral_w1_summary
from graph_fans.phase2.trainer import Trainer, TrainConfig
from graph_fans.phase6.augmentation import spectral_augmentation
from graph_fans.utils.graph_generators import generate_sbm
from graph_fans.utils.multiscale_features import generate_community_boundary_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

N_NODES = 50
N_FEATURES = 4
N_TRAIN = 100
N_REF = 50
N_GEN = 50
B = 8


def run_seed(
    graph,
    evecs: np.ndarray,
    bands: list[np.ndarray],
    seed: int,
    n_epochs: int,
    device: str,
    sigma: float = 0.3,
) -> dict:
    """Run one seed: generate 1 feature matrix, spectral-augment, train uniform vs band."""
    # Generate ONE community feature matrix (the seed sample)
    seed_features = generate_community_boundary_features(
        graph, n_features=N_FEATURES, seed=seed,
    )

    # Spectral-augment to N_TRAIN samples
    train_dataset = spectral_augmentation(
        seed_features, evecs, n_samples=N_TRAIN, sigma=sigma, seed=seed,
    ).astype(np.float32)

    # Reference split from a different base seed (no train/ref leakage)
    ref_seed_features = generate_community_boundary_features(
        graph, n_features=N_FEATURES, seed=seed + 100000,
    )
    ref_dataset = spectral_augmentation(
        ref_seed_features, evecs, n_samples=N_REF, sigma=sigma, seed=seed + 100000,
    ).astype(np.float32)

    # Compute importance weights from AUGMENTED TRAINING data only
    # (matches how Phase 2g computed weights — from the training split)
    profiles = []
    for feat in train_dataset[:50]:
        profiles.append(compute_band_energy(feat, evecs, bands))
    weights = compute_importance_weights(np.mean(profiles, axis=0))

    seed_entry = {"seed": seed}
    results = {}

    for method, use_spectral in [("uniform", False), ("spectral", True)]:
        cfg = TrainConfig(
            n_epochs=n_epochs,
            batch_timesteps=32,
            seed=seed,
            device=device,
            hidden_dim=128,
            n_layers=3,
            conv_type="gcn",
            sde_type="cosine",
            use_ema=True,
            use_lr_scheduler=True,
            n_train_samples=N_TRAIN,
            use_spectral_noise=use_spectral,
            B=B,
        )
        trainer = Trainer(
            cfg, graph, train_dataset,
            importance_weights=weights if use_spectral else None,
        )
        history = trainer.train()
        gen = trainer.generate(n_samples=N_GEN)
        sanity = trainer.sanity_check(train_dataset, n_gen=5)

        w1 = spectral_w1_summary(ref_dataset, gen, evecs, bands)

        results[method] = {
            "final_loss": float(history["loss"][-1]),
            "std_ratio": float(sanity["std_ratio"]),
            "w1_total": float(w1["total_w1"]),
            "w1_low": float(w1["low_band_w1"]),
            "w1_high": float(w1["high_band_w1"]),
            "per_band_w1": w1["per_band_w1"].tolist(),
        }

        logger.info(
            f"  seed={seed}/{method}: "
            f"loss={history['loss'][-1]:.3f}, "
            f"std_ratio={sanity['std_ratio']:.2f}, "
            f"W1={w1['total_w1']:.1f}"
        )

    seed_entry.update(results)
    return seed_entry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6e: synthetic spectral augmentation diagnostic"
    )
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sigma", type=float, default=0.3,
                        help="Spectral augmentation noise scale")
    parser.add_argument("--output", default="results/phase6e/synthetic_spectral_aug.json")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Phase 6e: SBM(q=0.05), n={N_NODES}, d={N_FEATURES}, sigma={args.sigma}")
    logger.info(f"Testing band shaping with spectral-augmented synthetic features")
    logger.info(f"Comparison baseline: Phase 2g = 12.5% W1 improvement (independent samples)")

    # Fixed graph topology (same as Phase 2g)
    gd = generate_sbm(
        n_nodes=N_NODES, p_inter=0.05, seed=0, feature_mode="community",
    )
    graph = gd.graph
    evals, evecs = compute_laplacian_spectrum(graph)
    _, bands = partition_into_bands(evals, B=B)

    per_seed = []
    uniform_w1s = []
    spectral_w1s = []

    for seed in range(args.n_seeds):
        entry = run_seed(
            graph, evecs, bands, seed, args.n_epochs, args.device, sigma=args.sigma,
        )
        per_seed.append(entry)
        uniform_w1s.append(entry["uniform"]["w1_total"])
        spectral_w1s.append(entry["spectral"]["w1_total"])

    u = np.array(uniform_w1s)
    s = np.array(spectral_w1s)
    diff = u - s

    if len(u) >= 2:
        t_stat, p_val = ttest_rel(u, s)
    else:
        t_stat, p_val = 0.0, 1.0

    improvement_pct = float(diff.mean() / u.mean() * 100) if u.mean() > 0 else 0.0
    significant = bool(p_val < 0.05)

    logger.info("")
    logger.info("=" * 80)
    logger.info("Phase 6e SUMMARY")
    logger.info("=" * 80)
    logger.info(f"  Uniform W1:  {u.mean():.1f} +/- {u.std():.1f}")
    logger.info(f"  Spectral W1: {s.mean():.1f} +/- {s.std():.1f}")
    logger.info(f"  Improvement: {diff.mean():.1f} ({improvement_pct:.1f}%)")
    logger.info(f"  t={t_stat:.3f}, p={p_val:.4f}")
    logger.info(f"  Significant (alpha=0.05): {significant}")
    logger.info("")
    logger.info("  GATE: " + ("GO — H1 REJECTED (shaping still helps)" if significant and improvement_pct > 0
                              else "NO-GO — H1 CONFIRMED (spectral augmentation eliminates shaping effect)"))

    result = {
        "experiment": "phase6e_synthetic_spectral_aug",
        "graph_family": "SBM(q=0.05)",
        "n_nodes": N_NODES,
        "n_features": N_FEATURES,
        "n_seeds": args.n_seeds,
        "n_epochs": args.n_epochs,
        "sigma": args.sigma,
        "uniform_w1_mean": float(u.mean()),
        "uniform_w1_std": float(u.std()),
        "spectral_w1_mean": float(s.mean()),
        "spectral_w1_std": float(s.std()),
        "improvement_pct": improvement_pct,
        "t_stat": float(t_stat),
        "p_val": float(p_val),
        "significant": significant,
        "phase2g_baseline_improvement_pct": 12.5,
        "per_seed": per_seed,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
