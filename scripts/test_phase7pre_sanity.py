"""Phase 7-pre: Continuous-feature sanity check.

Loads Elliptic Bitcoin and Stanford PPI, samples BFS subgraphs at n=100, reduces
features via TruncatedSVD to d=16, trains a baseline 3L GCN with uniform noise +
gaussian augmentation for a representative subset of subgraphs, and reports:

  - SVD variance explained at d=16
  - Spectral energy ratio (max/min band energy)
  - Per-subgraph std_ratio, spectral_L2, W1

Gate: both datasets load AND GCN denoises on >= 2/3 sanity subgraphs per dataset
(std_ratio < 3.0). Emits a JSON to results/phase7pre/ with gate decision.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from graph_fans.phase0.spectral_profiler import (
    compute_band_energy,
    compute_energy_ratio,
    compute_laplacian_spectrum,
    partition_into_bands,
)
from graph_fans.phase2.spectral_wasserstein import spectral_w1_summary
from graph_fans.phase2.trainer import Trainer, TrainConfig
from graph_fans.phase6.augmentation import gaussian_augmentation
from graph_fans.phase6.feature_reducer import fit_feature_reducer, reduce_features
from graph_fans.phase6.subgraph_sampler import sample_multiple_subgraphs
from graph_fans.phase7 import load_real_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

N_NODES = 100
N_DIMS = 16
N_AUG = 100
N_GEN = 50
B = 8


def _run_baseline(sample, sub_features, evals, evecs, bands, n_epochs, device, seed):
    train_data = gaussian_augmentation(sub_features, n_samples=N_AUG, seed=seed)
    ref_data = sub_features[np.newaxis, ...]

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
        n_train_samples=N_AUG,
        noise_shaping="uniform",
    )
    trainer = Trainer(cfg, sample.graph, train_data)
    history = trainer.train()
    sanity = trainer.sanity_check(train_data, n_gen=5)
    gen_data = trainer.generate(n_samples=N_GEN)
    w1 = spectral_w1_summary(ref_data, gen_data, evecs, bands)

    return {
        "final_loss": float(history["loss"][-1]),
        "std_ratio": float(sanity["std_ratio"]),
        "spectral_l2": float(sanity["spectral_l2"]),
        "w1_total": float(w1["total_w1"]),
        "training_var": float(train_data.var()),
        "w1_vs_train_var_ratio": float(w1["total_w1"] / (train_data.var() + 1e-12)),
    }


def _process_dataset(
    dataset_name: str,
    n_subgraphs: int,
    n_epochs: int,
    n_train_subgraphs: int,
    device: str,
) -> dict:
    logger.info("\n%s", "=" * 70)
    logger.info("Loading %s...", dataset_name)
    logger.info("%s", "=" * 70)

    load_kwargs = {}
    if dataset_name == "ppi":
        load_kwargs = {"split": "train", "concatenate": True}

    t0 = time.time()
    try:
        gd = load_real_dataset(dataset_name, **load_kwargs)
    except Exception as e:
        logger.error("Failed to load %s: %s", dataset_name, e)
        return {
            "dataset": dataset_name,
            "loaded": False,
            "error": str(e),
            "gate_pass": False,
        }
    logger.info(
        "%s loaded in %.1fs: %d nodes, %d edges, %d features",
        gd.name,
        time.time() - t0,
        gd.graph.number_of_nodes(),
        gd.graph.number_of_edges(),
        gd.features.shape[1],
    )

    # Fit SVD on full features
    full_features = gd.features.astype(np.float64)
    t0 = time.time()
    reducer = fit_feature_reducer(full_features, n_components=N_DIMS, method="truncated_svd", seed=0)
    variance_explained = float(reducer.explained_variance_ratio_.sum())
    logger.info("SVD d=%d: variance explained %.1f%% (%.1fs)", N_DIMS, 100 * variance_explained, time.time() - t0)

    # Sample subgraphs
    logger.info("Sampling %d BFS subgraphs at n=%d...", n_subgraphs, N_NODES)
    subgraphs = sample_multiple_subgraphs(gd.graph, N_NODES, k=n_subgraphs, base_seed=0)
    logger.info("Sampled %d subgraphs", len(subgraphs))

    per_subgraph: list[dict] = []
    train_results: list[dict] = []

    for i, sample in enumerate(subgraphs):
        sub_features = reduce_features(full_features[sample.node_indices], reducer)
        evals, evecs = compute_laplacian_spectrum(sample.graph)
        _, bands = partition_into_bands(evals, B=B)
        band_energy = compute_band_energy(sub_features, evecs, bands)
        energy_ratio = float(compute_energy_ratio(band_energy))

        entry = {
            "subgraph_idx": i,
            "n_nodes": sample.graph.number_of_nodes(),
            "energy_ratio": energy_ratio,
            "feat_std": float(sub_features.std()),
            "feat_min": float(sub_features.min()),
            "feat_max": float(sub_features.max()),
        }
        per_subgraph.append(entry)

        logger.info(
            "  subgraph %d/%d: n=%d, energy_ratio=%.2f",
            i + 1, len(subgraphs), sample.graph.number_of_nodes(), energy_ratio,
        )

        # Train on first n_train_subgraphs subgraphs
        if i < n_train_subgraphs:
            t0 = time.time()
            res = _run_baseline(sample, sub_features, evals, evecs, bands, n_epochs, device, seed=i)
            res["subgraph_idx"] = i
            res["energy_ratio"] = energy_ratio
            res["elapsed_s"] = time.time() - t0
            res["passed"] = bool(res["std_ratio"] < 3.0)
            train_results.append(res)
            logger.info(
                "    train: std_ratio=%.2f W1=%.1f loss=%.4f %s (%.1fs)",
                res["std_ratio"], res["w1_total"], res["final_loss"],
                "PASS" if res["passed"] else "FAIL", res["elapsed_s"],
            )

    n_pass = sum(1 for r in train_results if r["passed"])
    pass_rate = n_pass / max(len(train_results), 1)
    gate_pass = pass_rate >= 2.0 / 3.0

    logger.info(
        "%s: %d/%d training passes (std_ratio<3.0), gate=%s",
        dataset_name, n_pass, len(train_results), "PASS" if gate_pass else "FAIL",
    )

    return {
        "dataset": dataset_name,
        "loaded": True,
        "n_nodes_full": int(gd.graph.number_of_nodes()),
        "n_edges_full": int(gd.graph.number_of_edges()),
        "n_features_full": int(full_features.shape[1]),
        "svd_variance_explained_d16": variance_explained,
        "n_subgraphs": len(subgraphs),
        "n_train_subgraphs": len(train_results),
        "per_subgraph": per_subgraph,
        "train_results": train_results,
        "n_pass": n_pass,
        "pass_rate": pass_rate,
        "gate_pass": gate_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7-pre sanity check")
    parser.add_argument("--datasets", type=str, default="elliptic,ppi",
                        help="Comma-separated dataset names")
    parser.add_argument("--n-subgraphs", type=int, default=10,
                        help="Subgraphs per dataset for spectral profiling")
    parser.add_argument("--n-train-subgraphs", type=int, default=3,
                        help="Number of subgraphs to train baseline on")
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="results/phase7pre/sanity_results.json")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 3 subgraphs, 50 epochs, 2 training")
    args = parser.parse_args()

    if args.quick:
        args.n_subgraphs = 3
        args.n_train_subgraphs = 2
        args.n_epochs = 50

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    all_results: dict = {
        "experiment": "phase7pre_sanity",
        "config": {
            "datasets": datasets,
            "n_subgraphs": args.n_subgraphs,
            "n_train_subgraphs": args.n_train_subgraphs,
            "n_epochs": args.n_epochs,
            "n_nodes": N_NODES,
            "n_dims": N_DIMS,
            "device": args.device,
            "quick": args.quick,
        },
        "datasets": {},
    }

    for dataset in datasets:
        all_results["datasets"][dataset] = _process_dataset(
            dataset_name=dataset,
            n_subgraphs=args.n_subgraphs,
            n_epochs=args.n_epochs,
            n_train_subgraphs=args.n_train_subgraphs,
            device=args.device,
        )
        # Save incrementally
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)

    all_results["total_elapsed_s"] = time.time() - t_start

    # Overall gate
    loaded_ok = all(all_results["datasets"][d].get("loaded", False) for d in datasets)
    per_dataset_gates = [all_results["datasets"][d].get("gate_pass", False) for d in datasets]
    overall_gate = loaded_ok and all(per_dataset_gates)
    all_results["overall_gate_pass"] = overall_gate

    print("\n" + "=" * 70)
    print("PHASE 7-PRE SUMMARY")
    print("=" * 70)
    for d in datasets:
        r = all_results["datasets"][d]
        if not r.get("loaded", False):
            print(f"  {d}: FAILED TO LOAD — {r.get('error', '?')}")
            continue
        print(
            f"  {d}: {r['n_nodes_full']} nodes, "
            f"SVD var={r['svd_variance_explained_d16']:.1%}, "
            f"train pass {r['n_pass']}/{r['n_train_subgraphs']} "
            f"({'GO' if r['gate_pass'] else 'NO-GO'})"
        )
    print(f"\nOVERALL GATE: {'GO' if overall_gate else 'NO-GO'}")
    print(f"Total time: {all_results['total_elapsed_s']:.0f}s")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
