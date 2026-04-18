"""Phase 7a: Multi-dataset spectral augmentation validation.

For each dataset × augmentation × subgraph:
  1. Extract BFS subgraph at n=100, reduce features to d=16 via TruncatedSVD
     (reducer fit on the full dataset features).
  2. Create augmented training data (gaussian | spectral | dropout).
  3. Train a 3L GCN diffusion model (500 epochs, cosine+EMA, uniform noise).
  4. Generate samples and compute spectral W1 vs un-augmented real features.

Statistics: paired t-test across subgraphs comparing spectral vs gaussian and
spectral vs dropout.

Gate tiers:
  - Tier 1: spectral wins at p<0.05 on >= 3/4 datasets → proceed to 7b
  - Tier 2: wins on >= 2 datasets → proceed with caveats
  - Tier 3: only Cora → stop pipeline
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

from graph_fans.phase0.spectral_profiler import (
    compute_band_energy,
    compute_energy_ratio,
    compute_laplacian_spectrum,
    partition_into_bands,
)
from graph_fans.phase2.spectral_wasserstein import spectral_w1_summary
from graph_fans.phase2.trainer import Trainer, TrainConfig
from graph_fans.phase6.augmentation import (
    dropout_augmentation,
    gaussian_augmentation,
    spectral_augmentation,
)
from graph_fans.phase6.feature_reducer import fit_feature_reducer, reduce_features
from graph_fans.phase6.subgraph_sampler import sample_multiple_subgraphs
from graph_fans.phase7 import load_real_dataset
from graph_fans.utils.graph_generators import load_citation_network

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

N_NODES = 100
N_DIMS = 16
N_AUG = 100
N_GEN = 50
B = 8

AUGMENTATIONS = ["gaussian", "spectral", "dropout"]


def _build_aug(strategy: str, features, evecs, n_samples, seed):
    if strategy == "gaussian":
        return gaussian_augmentation(features, n_samples=n_samples, seed=seed)
    if strategy == "spectral":
        return spectral_augmentation(features, evecs, n_samples=n_samples, sigma=0.3, seed=seed)
    if strategy == "dropout":
        return dropout_augmentation(features, n_samples=n_samples, drop_rate=0.2, seed=seed)
    raise ValueError(f"Unknown augmentation {strategy!r}")


def _load_dataset(name: str):
    key = name.strip().lower()
    if key in {"cora", "citeseer"}:
        gd = load_citation_network(name.capitalize())
        return gd
    return load_real_dataset(name)


def _run_cell(sample, sub_features, evals, evecs, bands, strategy, n_epochs, device, subgraph_idx):
    t0 = time.time()
    train_data = _build_aug(strategy, sub_features, evecs, N_AUG, seed=subgraph_idx)
    ref_data = sub_features[np.newaxis, ...]

    cfg = TrainConfig(
        n_epochs=n_epochs,
        batch_timesteps=32,
        seed=subgraph_idx,
        device=device,
        hidden_dim=128,
        n_layers=3,
        conv_type="gcn",
        sde_type="cosine",
        use_ema=True,
        use_lr_scheduler=True,
        n_train_samples=N_AUG,
        noise_shaping="uniform",
    )
    trainer = Trainer(cfg, sample.graph, train_data)
    history = trainer.train()
    sanity = trainer.sanity_check(train_data, n_gen=5)
    gen_data = trainer.generate(n_samples=N_GEN)
    w1 = spectral_w1_summary(ref_data, gen_data, evecs, bands)
    elapsed = time.time() - t0
    return {
        "subgraph_idx": subgraph_idx,
        "strategy": strategy,
        "final_loss": float(history["loss"][-1]),
        "std_ratio": float(sanity["std_ratio"]),
        "spectral_l2": float(sanity["spectral_l2"]),
        "w1_total": float(w1["total_w1"]),
        "w1_low": float(w1["low_band_w1"]),
        "w1_high": float(w1["high_band_w1"]),
        "elapsed_s": elapsed,
    }


def _process_dataset(name: str, n_subgraphs: int, n_epochs: int, device: str):
    logger.info("\n%s", "=" * 70)
    logger.info("Dataset: %s", name)
    logger.info("%s", "=" * 70)

    gd = _load_dataset(name)
    full_features = gd.features.astype(np.float64)

    # Feature properties
    feat_sparsity = float((full_features != 0).mean())

    reducer = fit_feature_reducer(full_features, n_components=N_DIMS, method="truncated_svd", seed=0)
    variance_explained = float(reducer.explained_variance_ratio_.sum())

    subgraphs = sample_multiple_subgraphs(gd.graph, N_NODES, k=n_subgraphs, base_seed=0)
    logger.info("%s: %d nodes, feat_sparsity=%.3f, SVD d=%d var=%.1f%%, %d subgraphs",
                name, gd.graph.number_of_nodes(), feat_sparsity, N_DIMS, 100 * variance_explained, len(subgraphs))

    per_subgraph_results: list[dict] = []
    energy_ratios: list[float] = []

    for i, sample in enumerate(subgraphs):
        sub_features = reduce_features(full_features[sample.node_indices], reducer)
        evals, evecs = compute_laplacian_spectrum(sample.graph)
        _, bands = partition_into_bands(evals, B=B)
        be = compute_band_energy(sub_features, evecs, bands)
        er = float(compute_energy_ratio(be))
        energy_ratios.append(er)

        entry = {"subgraph_idx": i, "energy_ratio": er, "cells": {}}
        logger.info("  subgraph %d/%d energy_ratio=%.2f", i + 1, len(subgraphs), er)

        for strategy in AUGMENTATIONS:
            cell = _run_cell(sample, sub_features, evals, evecs, bands, strategy, n_epochs, device, i)
            entry["cells"][strategy] = cell
            logger.info(
                "    %s: std_ratio=%.2f W1=%.1f (%.1fs)",
                strategy, cell["std_ratio"], cell["w1_total"], cell["elapsed_s"],
            )
        per_subgraph_results.append(entry)

    # Paired t-tests
    w1_by_strategy = {
        s: np.array([e["cells"][s]["w1_total"] for e in per_subgraph_results])
        for s in AUGMENTATIONS
    }

    def _paired(a: str, b: str) -> dict:
        x = w1_by_strategy[a]
        y = w1_by_strategy[b]
        diff = x - y  # positive => a has higher W1 (worse) than b
        t, p = ttest_rel(x, y)
        return {
            "a": a,
            "b": b,
            "a_mean": float(np.mean(x)),
            "b_mean": float(np.mean(y)),
            "a_std": float(np.std(x)),
            "b_std": float(np.std(y)),
            "mean_diff_a_minus_b": float(np.mean(diff)),
            "improvement_pct_b_vs_a": float(100 * (np.mean(x) - np.mean(y)) / np.mean(x)) if np.mean(x) > 0 else 0.0,
            "t_stat": float(t),
            "p_val": float(p),
            "n": int(len(diff)),
        }

    tests = {
        "spectral_vs_gaussian": _paired("gaussian", "spectral"),
        "spectral_vs_dropout": _paired("dropout", "spectral"),
        "dropout_vs_gaussian": _paired("gaussian", "dropout"),
    }

    # "spectral wins" = spectral W1 mean is lower than gaussian AND p<0.05
    sp = tests["spectral_vs_gaussian"]
    spectral_beats_gaussian = bool(sp["b_mean"] < sp["a_mean"] and sp["p_val"] < 0.05)

    result = {
        "dataset": name,
        "n_nodes_full": int(gd.graph.number_of_nodes()),
        "n_edges_full": int(gd.graph.number_of_edges()),
        "n_features_full": int(full_features.shape[1]),
        "feat_sparsity": feat_sparsity,
        "svd_variance_explained_d16": variance_explained,
        "energy_ratio_mean": float(np.mean(energy_ratios)),
        "energy_ratio_std": float(np.std(energy_ratios)),
        "n_subgraphs": len(subgraphs),
        "per_subgraph": per_subgraph_results,
        "tests": tests,
        "spectral_beats_gaussian": spectral_beats_gaussian,
    }
    logger.info(
        "%s: gaussian=%.1f±%.1f spectral=%.1f±%.1f dropout=%.1f±%.1f  spectral vs gaussian p=%.4f (%s)",
        name,
        tests["spectral_vs_gaussian"]["a_mean"], tests["spectral_vs_gaussian"]["a_std"],
        tests["spectral_vs_gaussian"]["b_mean"], tests["spectral_vs_gaussian"]["b_std"],
        tests["spectral_vs_dropout"]["a_mean"], tests["spectral_vs_dropout"]["a_std"],
        sp["p_val"], "WIN" if spectral_beats_gaussian else "no-win",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7a multi-dataset spectral augmentation")
    parser.add_argument("--datasets", type=str, default="cora,elliptic,ppi")
    parser.add_argument("--n-subgraphs", type=int, default=30)
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="results/phase7a/multidataset_results.json")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.n_subgraphs = 3
        args.n_epochs = 50

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    all_results = {
        "experiment": "phase7a_multidataset",
        "config": {
            "datasets": datasets,
            "n_subgraphs": args.n_subgraphs,
            "n_epochs": args.n_epochs,
            "n_nodes": N_NODES,
            "n_dims": N_DIMS,
            "device": args.device,
            "quick": args.quick,
        },
        "datasets": {},
    }
    t_start = time.time()
    for d in datasets:
        try:
            all_results["datasets"][d] = _process_dataset(d, args.n_subgraphs, args.n_epochs, args.device)
        except Exception as e:
            logger.exception("Failed on dataset %s", d)
            all_results["datasets"][d] = {"error": str(e)}
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)

    all_results["total_elapsed_s"] = time.time() - t_start

    # Gate tier
    wins = sum(
        1 for d in datasets
        if isinstance(all_results["datasets"][d], dict) and all_results["datasets"][d].get("spectral_beats_gaussian", False)
    )
    n = len(datasets)
    if n >= 4 and wins >= 3:
        tier = "Tier 1"
    elif wins >= 2:
        tier = "Tier 2"
    elif wins == 1:
        tier = "Tier 2 (single-dataset)" if "cora" not in datasets else "Tier 3"
    else:
        tier = "Tier 3"

    all_results["gate_tier"] = tier
    all_results["n_wins"] = wins

    print("\n" + "=" * 70)
    print("PHASE 7a SUMMARY")
    print("=" * 70)
    for d in datasets:
        r = all_results["datasets"][d]
        if "error" in r:
            print(f"  {d}: ERROR — {r['error']}")
            continue
        sp = r["tests"]["spectral_vs_gaussian"]
        print(
            f"  {d:10s}: gaussian={sp['a_mean']:.1f} spectral={sp['b_mean']:.1f} "
            f"(delta={sp['improvement_pct_b_vs_a']:+.1f}%, p={sp['p_val']:.4f}) "
            f"{'WIN' if r['spectral_beats_gaussian'] else '-'}"
        )
    print(f"\nOverall: {wins}/{n} datasets; {tier}")
    print(f"Total: {all_results['total_elapsed_s']:.0f}s")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
