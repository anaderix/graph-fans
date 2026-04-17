"""Phase 6f: Cross-protocol 2x2x2 evaluation of noise shaping.

Phase 2g showed band shaping gives +12.5% W1 improvement (p=0.0001) on
SBM(q=0.05) with 100 INDEPENDENT feature samples for training and 50
INDEPENDENT samples for evaluation reference.

Phase 6e replaced BOTH the training dataset and the evaluation reference with
spectral-augmented-from-1-sample data and observed -2% (p=0.629).

Phase 6f disambiguates: does the training-data change, the evaluation-ref
change, or both kill the shaping effect? For each seed we train 4 models
(2 training protocols x 2 shaping methods) and evaluate each against both
reference sets (2 reference protocols) => 8 W1 numbers per seed.

Cells:
    A_uni  : indep train   + uniform shaping + indep ref
    A_band : indep train   + band    shaping + indep ref
    B_uni  : indep train   + uniform shaping + specaug ref
    B_band : indep train   + band    shaping + specaug ref
    C_uni  : specaug train + uniform shaping + indep ref
    C_band : specaug train + band    shaping + indep ref
    D_uni  : specaug train + uniform shaping + specaug ref
    D_band : specaug train + band    shaping + specaug ref

Key comparisons:
    A_band vs A_uni: replicates Phase 2g (~+12.5% expected)
    B_band vs B_uni: indep training + specaug ref (isolates eval protocol)
    C_band vs C_uni: specaug training + indep ref (isolates train protocol)
    D_band vs D_uni: replicates Phase 6e (~flat expected)
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
from graph_fans.utils.multiscale_features import (
    generate_community_boundary_features,
    generate_feature_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

N_NODES = 50
N_FEATURES = 4
N_TRAIN = 100
N_REF = 50
N_GEN = 50
B = 8
SPECAUG_SIGMA = 0.3


def generate_independent_set(
    graph,
    n_samples: int,
    base_seed: int,
) -> np.ndarray:
    """Generate independent community-boundary features (Phase 2g protocol)."""
    return generate_feature_dataset(
        graph,
        n_samples=n_samples,
        n_features=N_FEATURES,
        base_seed=base_seed,
        mode="community",
    ).astype(np.float32)


def generate_specaug_set(
    graph,
    evecs: np.ndarray,
    n_samples: int,
    seed: int,
    aug_seed: int,
) -> np.ndarray:
    """Generate spectral-augmented features from one seed matrix (Phase 6e protocol)."""
    seed_features = generate_community_boundary_features(
        graph, n_features=N_FEATURES, seed=seed,
    )
    return spectral_augmentation(
        seed_features, evecs, n_samples=n_samples, sigma=SPECAUG_SIGMA, seed=aug_seed,
    ).astype(np.float32)


def compute_weights_from_train(
    train_dataset: np.ndarray,
    evecs: np.ndarray,
    bands: list[np.ndarray],
):
    """Compute importance weights from first 50 samples of training set (no ref leakage)."""
    profiles = []
    for feat in train_dataset[: min(50, len(train_dataset))]:
        profiles.append(compute_band_energy(feat, evecs, bands))
    return compute_importance_weights(np.mean(profiles, axis=0))


def train_and_generate(
    cfg: TrainConfig,
    graph,
    train_dataset: np.ndarray,
    weights,
) -> tuple[np.ndarray, dict]:
    """Train one model, return generated samples and sanity info."""
    trainer = Trainer(
        cfg, graph, train_dataset,
        importance_weights=weights if cfg.use_spectral_noise else None,
    )
    history = trainer.train()
    gen = trainer.generate(n_samples=N_GEN)
    sanity = trainer.sanity_check(train_dataset, n_gen=5)
    info = {
        "final_loss": float(history["loss"][-1]),
        "std_ratio": float(sanity["std_ratio"]),
    }
    return gen, info


def evaluate_against(
    gen: np.ndarray,
    ref: np.ndarray,
    evecs: np.ndarray,
    bands: list[np.ndarray],
) -> dict:
    """Compute W1 summary of gen vs ref."""
    w1 = spectral_w1_summary(ref, gen, evecs, bands)
    return {
        "w1_total": float(w1["total_w1"]),
        "w1_low": float(w1["low_band_w1"]),
        "w1_high": float(w1["high_band_w1"]),
        "per_band_w1": w1["per_band_w1"].tolist(),
    }


def run_seed(
    graph,
    evecs: np.ndarray,
    bands: list[np.ndarray],
    seed: int,
    n_epochs: int,
    device: str,
) -> dict:
    """Run one seed: 4 models x 2 references = 8 W1 numbers.

    Seed-offset scheme (independent of other seeds to avoid collisions):
    - indep_train_seed = seed * 10000
    - indep_ref_seed   = seed * 10000 + 100000
    - specaug_train_src_seed = seed * 10000 + 200000  (the 1 sample used)
    - specaug_train_aug_seed = seed * 10000 + 200001
    - specaug_ref_src_seed   = seed * 10000 + 300000
    - specaug_ref_aug_seed   = seed * 10000 + 300001
    """
    logger.info(f"\n[seed={seed}] generating datasets...")

    indep_train = generate_independent_set(
        graph, N_TRAIN, base_seed=seed * 10000,
    )
    indep_ref = generate_independent_set(
        graph, N_REF, base_seed=seed * 10000 + 100000,
    )
    specaug_train = generate_specaug_set(
        graph, evecs, N_TRAIN,
        seed=seed * 10000 + 200000,
        aug_seed=seed * 10000 + 200001,
    )
    specaug_ref = generate_specaug_set(
        graph, evecs, N_REF,
        seed=seed * 10000 + 300000,
        aug_seed=seed * 10000 + 300001,
    )

    indep_weights = compute_weights_from_train(indep_train, evecs, bands)
    specaug_weights = compute_weights_from_train(specaug_train, evecs, bands)

    logger.info(
        f"[seed={seed}] indep weights: {indep_weights.weights.round(3)}"
    )
    logger.info(
        f"[seed={seed}] specaug weights: {specaug_weights.weights.round(3)}"
    )

    # Train 4 models
    models = []  # list of (train_protocol, shaping, gen, info)
    for train_name, train_ds, weights in [
        ("indep", indep_train, indep_weights),
        ("specaug", specaug_train, specaug_weights),
    ]:
        for shape_name, use_spectral in [("uniform", False), ("band", True)]:
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
            gen, info = train_and_generate(cfg, graph, train_ds, weights)
            logger.info(
                f"[seed={seed}] trained {train_name}/{shape_name}: "
                f"loss={info['final_loss']:.3f}, std_ratio={info['std_ratio']:.2f}"
            )
            models.append((train_name, shape_name, gen, info))

    # Evaluate each model against both references
    cell_labels = {
        ("indep", "uniform"): ("A_uni", "A_uni_B", "B_uni"),
        ("indep", "band"): ("A_band", "A_band_B", "B_band"),
        ("specaug", "uniform"): ("C_uni", "C_uni_D", "D_uni"),
        ("specaug", "band"): ("C_band", "C_band_D", "D_band"),
    }
    # For clarity: each training+shape model -> (label_against_indep, label_against_specaug)
    label_map = {
        ("indep", "uniform"): ("A_uni", "B_uni"),
        ("indep", "band"): ("A_band", "B_band"),
        ("specaug", "uniform"): ("C_uni", "D_uni"),
        ("specaug", "band"): ("C_band", "D_band"),
    }

    cells: dict[str, dict] = {}
    for train_name, shape_name, gen, info in models:
        label_indep, label_specaug = label_map[(train_name, shape_name)]

        w1_indep = evaluate_against(gen, indep_ref, evecs, bands)
        w1_specaug = evaluate_against(gen, specaug_ref, evecs, bands)

        cells[label_indep] = {
            "train_protocol": train_name,
            "shaping": shape_name,
            "ref_protocol": "indep",
            "final_loss": info["final_loss"],
            "std_ratio": info["std_ratio"],
            **w1_indep,
        }
        cells[label_specaug] = {
            "train_protocol": train_name,
            "shaping": shape_name,
            "ref_protocol": "specaug",
            "final_loss": info["final_loss"],
            "std_ratio": info["std_ratio"],
            **w1_specaug,
        }

        logger.info(
            f"[seed={seed}] {train_name}/{shape_name}: "
            f"W1(indep_ref)={w1_indep['w1_total']:.1f}, "
            f"W1(specaug_ref)={w1_specaug['w1_total']:.1f}"
        )

    return {"seed": seed, "cells": cells}


def compute_comparison(
    per_seed: list[dict],
    uni_label: str,
    band_label: str,
) -> dict:
    """Compute paired t-test comparing band vs uniform for a given cell family."""
    uni_vals = np.array(
        [entry["cells"][uni_label]["w1_total"] for entry in per_seed]
    )
    band_vals = np.array(
        [entry["cells"][band_label]["w1_total"] for entry in per_seed]
    )
    diff = uni_vals - band_vals

    if len(uni_vals) >= 2:
        t_stat, p_val = ttest_rel(uni_vals, band_vals)
    else:
        t_stat, p_val = 0.0, 1.0

    improvement_pct = (
        float(diff.mean() / uni_vals.mean() * 100) if uni_vals.mean() > 0 else 0.0
    )

    return {
        "uni_label": uni_label,
        "band_label": band_label,
        "uniform_w1_mean": float(uni_vals.mean()),
        "uniform_w1_std": float(uni_vals.std()),
        "band_w1_mean": float(band_vals.mean()),
        "band_w1_std": float(band_vals.std()),
        "mean_delta_uni_minus_band": float(diff.mean()),
        "improvement_pct": improvement_pct,
        "t_stat": float(t_stat),
        "p_val": float(p_val),
        "significant": bool(p_val < 0.05),
        "band_better_count": int((diff > 0).sum()),
        "n_seeds": int(len(uni_vals)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6f: cross-protocol 2x2x2 shaping evaluation"
    )
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="results/phase6f/")
    parser.add_argument(
        "--quick", action="store_true",
        help="Debug mode: 2 seeds, 50 epochs.",
    )
    args = parser.parse_args()

    if args.quick:
        args.n_seeds = 2
        args.n_epochs = 50

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "crossprotocol_results.json"

    logger.info(
        f"Phase 6f: SBM(q=0.05), n={N_NODES}, d={N_FEATURES}, "
        f"n_seeds={args.n_seeds}, n_epochs={args.n_epochs}"
    )
    logger.info(
        f"Per seed: 4 training runs x 2 refs = 8 W1 cells. "
        f"Total runs: {args.n_seeds * 4} trainings."
    )

    # Fixed graph topology (same as Phase 2g / 6e)
    gd = generate_sbm(
        n_nodes=N_NODES, p_inter=0.05, seed=0, feature_mode="community",
    )
    graph = gd.graph
    evals, evecs = compute_laplacian_spectrum(graph)
    _, bands = partition_into_bands(evals, B=B)

    per_seed: list[dict] = []
    for seed in range(args.n_seeds):
        entry = run_seed(graph, evecs, bands, seed, args.n_epochs, args.device)
        per_seed.append(entry)

        # Crash-safe intermediate save
        with open(out_path, "w") as f:
            json.dump(
                {"status": "running", "seeds_completed": len(per_seed), "per_seed": per_seed},
                f,
                indent=2,
            )

    # Compute 4 key comparisons
    comparisons = {
        "A_band_vs_A_uni": compute_comparison(per_seed, "A_uni", "A_band"),
        "B_band_vs_B_uni": compute_comparison(per_seed, "B_uni", "B_band"),
        "C_band_vs_C_uni": compute_comparison(per_seed, "C_uni", "C_band"),
        "D_band_vs_D_uni": compute_comparison(per_seed, "D_uni", "D_band"),
    }

    # Also compute per-cell means
    cell_names = ["A_uni", "A_band", "B_uni", "B_band", "C_uni", "C_band", "D_uni", "D_band"]
    cell_summary = {}
    for name in cell_names:
        vals = np.array([entry["cells"][name]["w1_total"] for entry in per_seed])
        cell_summary[name] = {
            "w1_mean": float(vals.mean()),
            "w1_std": float(vals.std()),
            "n_seeds": int(len(vals)),
        }

    logger.info("")
    logger.info("=" * 80)
    logger.info("Phase 6f SUMMARY")
    logger.info("=" * 80)
    for name in cell_names:
        s = cell_summary[name]
        logger.info(f"  {name:<8}  W1 = {s['w1_mean']:>7.1f} +/- {s['w1_std']:>5.1f}")

    logger.info("")
    logger.info("Key comparisons (band vs uniform, paired t-test):")
    for key, comp in comparisons.items():
        logger.info(
            f"  {key:<22}  uniform={comp['uniform_w1_mean']:.1f}  "
            f"band={comp['band_w1_mean']:.1f}  "
            f"improv={comp['improvement_pct']:+.1f}%  "
            f"p={comp['p_val']:.4f}  "
            f"{'SIG' if comp['significant'] else '   '}"
        )

    result = {
        "experiment": "phase6f_crossprotocol",
        "graph_family": "SBM(q=0.05)",
        "n_nodes": N_NODES,
        "n_features": N_FEATURES,
        "n_train": N_TRAIN,
        "n_ref": N_REF,
        "n_gen": N_GEN,
        "n_seeds": args.n_seeds,
        "n_epochs": args.n_epochs,
        "specaug_sigma": SPECAUG_SIGMA,
        "phase2g_baseline_improvement_pct": 12.5,
        "phase6e_baseline_improvement_pct": -2.0,
        "cell_summary": cell_summary,
        "comparisons": comparisons,
        "per_seed": per_seed,
        "status": "complete",
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
