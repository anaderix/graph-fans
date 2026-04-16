"""Phase 6b: Augmentation strategy comparison on real Cora subgraphs.

Compares three augmentation strategies (gaussian, spectral, dropout) crossed
with two noise methods (uniform, band-shaped) on BFS subgraphs from Cora.

For each subgraph x augmentation x noise method:
  1. Extract and reduce features (TruncatedSVD d=16, fit on full Cora)
  2. Create augmented training data
  3. Compute importance weights from augmented data (not real features)
  4. Train diffusion model (500 epochs)
  5. Generate samples and evaluate W1 vs un-augmented real features

Gate: At least one augmentation produces non-degenerate features
(std_ratio < 3.0 for >= 75% of subgraphs).
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
from graph_fans.phase2.noise_shaper import compute_importance_weights
from graph_fans.phase2.spectral_wasserstein import spectral_w1_summary
from graph_fans.phase2.trainer import Trainer, TrainConfig
from graph_fans.phase6.augmentation import (
    gaussian_augmentation,
    spectral_augmentation,
    dropout_augmentation,
)
from graph_fans.phase6.feature_reducer import fit_feature_reducer, reduce_features
from graph_fans.phase6.subgraph_sampler import sample_multiple_subgraphs
from graph_fans.utils.graph_generators import load_citation_network

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

N_GEN = 50
N_AUG = 100
N_NODES = 100
N_DIMS = 16
B = 8

AUGMENTATION_STRATEGIES = ["gaussian", "spectral", "dropout"]
NOISE_METHODS = ["uniform", "band"]


def create_augmented_data(
    strategy: str,
    features: np.ndarray,
    eigenvectors: np.ndarray,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    if strategy == "gaussian":
        return gaussian_augmentation(features, n_samples=n_samples, seed=seed)
    elif strategy == "spectral":
        return spectral_augmentation(
            features, eigenvectors, n_samples=n_samples, sigma=0.3, seed=seed,
        )
    elif strategy == "dropout":
        return dropout_augmentation(
            features, n_samples=n_samples, drop_rate=0.2, seed=seed,
        )
    else:
        raise ValueError(f"Unknown augmentation strategy: {strategy}")


def run_single_cell(
    sample,
    sub_features: np.ndarray,
    evals: np.ndarray,
    evecs: np.ndarray,
    bands: list[np.ndarray],
    strategy: str,
    noise_method: str,
    n_epochs: int,
    device: str,
    subgraph_idx: int,
) -> dict:
    """Run one (subgraph, augmentation, noise_method) cell."""
    t0 = time.time()

    train_data = create_augmented_data(
        strategy, sub_features, evecs, n_samples=N_AUG, seed=subgraph_idx,
    )

    importance_weights = None
    if noise_method == "band":
        mean_band_energy = np.zeros(len(bands))
        for s in range(min(train_data.shape[0], 50)):
            mean_band_energy += compute_band_energy(train_data[s], evecs, bands)
        mean_band_energy /= min(train_data.shape[0], 50)
        importance_weights = compute_importance_weights(mean_band_energy)

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
        noise_shaping=noise_method,
    )
    trainer = Trainer(
        cfg, sample.graph, train_data,
        importance_weights=importance_weights,
    )
    history = trainer.train()

    sanity = trainer.sanity_check(train_data, n_gen=5)
    std_ratio = sanity["std_ratio"]

    gen_data = trainer.generate(n_samples=N_GEN)
    w1 = spectral_w1_summary(ref_data, gen_data, evecs, bands)

    elapsed = time.time() - t0
    passed = std_ratio < 3.0

    logger.info(
        f"    {strategy}/{noise_method}: "
        f"loss={history['loss'][-1]:.4f}, "
        f"std_ratio={std_ratio:.2f}, "
        f"W1={w1['total_w1']:.1f}, "
        f"{'PASS' if passed else 'FAIL'} "
        f"({elapsed:.1f}s)"
    )

    return {
        "subgraph_idx": subgraph_idx,
        "strategy": strategy,
        "noise_method": noise_method,
        "final_loss": float(history["loss"][-1]),
        "std_ratio": std_ratio,
        "spectral_l2": sanity["spectral_l2"],
        "w1_total": float(w1["total_w1"]),
        "w1_low": float(w1["low_band_w1"]),
        "w1_high": float(w1["high_band_w1"]),
        "per_band_w1": w1["per_band_w1"].tolist(),
        "passed": passed,
        "warnings": sanity["warnings"],
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6b: Augmentation strategy comparison on Cora subgraphs"
    )
    parser.add_argument(
        "--n-subgraphs", type=int, default=20,
        help="Number of BFS subgraphs (default: 20)",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=500,
        help="Training epochs per run (default: 500)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device (default: cuda)",
    )
    parser.add_argument(
        "--output", type=str, default="results/phase6b/augmentation_comparison.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 3 subgraphs, 50 epochs",
    )
    args = parser.parse_args()

    n_subgraphs = 3 if args.quick else args.n_subgraphs
    n_epochs = 50 if args.quick else args.n_epochs

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Cora citation network...")
    cora = load_citation_network("Cora")
    full_features = cora.features.astype(np.float64)
    logger.info(
        f"  Cora: {cora.graph.number_of_nodes()} nodes, "
        f"{cora.graph.number_of_edges()} edges, "
        f"features shape {full_features.shape}"
    )

    logger.info(f"Fitting TruncatedSVD to d={N_DIMS}...")
    reducer = fit_feature_reducer(
        full_features, n_components=N_DIMS, method="truncated_svd", seed=0,
    )
    variance_explained = reducer.explained_variance_ratio_.sum()
    logger.info(f"  d={N_DIMS}: {variance_explained:.1%} variance explained")

    logger.info(f"Sampling {n_subgraphs} BFS subgraphs at n={N_NODES}...")
    subgraphs = sample_multiple_subgraphs(
        cora.graph, N_NODES, k=n_subgraphs, base_seed=0,
    )

    all_results: list[dict] = []
    total_start = time.time()

    for i, sample in enumerate(subgraphs):
        sub_features_full = full_features[sample.node_indices]
        sub_features = reduce_features(sub_features_full, reducer)

        evals, evecs = compute_laplacian_spectrum(sample.graph)
        _, bands = partition_into_bands(evals, B=B)
        band_energy = compute_band_energy(sub_features, evecs, bands)
        energy_ratio = compute_energy_ratio(band_energy)

        logger.info(
            f"\nSubgraph {i+1}/{n_subgraphs} "
            f"(n={sample.graph.number_of_nodes()}, energy_ratio={energy_ratio:.1f})"
        )

        for strategy in AUGMENTATION_STRATEGIES:
            for noise_method in NOISE_METHODS:
                result = run_single_cell(
                    sample=sample,
                    sub_features=sub_features,
                    evals=evals,
                    evecs=evecs,
                    bands=bands,
                    strategy=strategy,
                    noise_method=noise_method,
                    n_epochs=n_epochs,
                    device=args.device,
                    subgraph_idx=i,
                )
                result["energy_ratio"] = energy_ratio
                all_results.append(result)

                output_data = {
                    "experiment": "phase6b_augmentation_comparison",
                    "dataset": "Cora",
                    "n_nodes": N_NODES,
                    "n_dims": N_DIMS,
                    "n_subgraphs": n_subgraphs,
                    "n_epochs": n_epochs,
                    "variance_explained": float(variance_explained),
                    "results": all_results,
                }
                with open(output_path, "w") as f:
                    json.dump(output_data, f, indent=2)

    total_elapsed = time.time() - total_start

    # Aggregate results per (strategy, noise_method)
    print(f"\n{'='*90}")
    print(f"Phase 6b Augmentation Comparison ({total_elapsed:.0f}s total)")
    print(f"{'='*90}")
    print(
        f"{'Strategy':>12} {'Noise':>8} {'Pass':>6} {'Rate':>6} "
        f"{'std_r':>7} {'W1':>10}"
    )
    print("-" * 90)

    gate_pass = False
    best_strategy = None
    best_w1 = float("inf")

    for strategy in AUGMENTATION_STRATEGIES:
        for noise_method in NOISE_METHODS:
            cells = [
                r for r in all_results
                if r["strategy"] == strategy and r["noise_method"] == noise_method
            ]
            n_pass = sum(1 for c in cells if c["passed"])
            n_total = len(cells)
            pass_rate = n_pass / max(n_total, 1)
            mean_std_r = np.mean([c["std_ratio"] for c in cells])
            mean_w1 = np.mean([c["w1_total"] for c in cells])
            std_w1 = np.std([c["w1_total"] for c in cells])

            rate_str = f"{n_pass}/{n_total}"
            print(
                f"{strategy:>12} {noise_method:>8} {rate_str:>6} "
                f"{pass_rate:>5.0%} "
                f"{mean_std_r:>7.2f} "
                f"{mean_w1:>7.1f}+/-{std_w1:<4.1f}"
            )

            if pass_rate >= 0.75:
                gate_pass = True
                if mean_w1 < best_w1:
                    best_w1 = mean_w1
                    best_strategy = f"{strategy}/{noise_method}"

    print(f"{'='*90}")
    print(f"\nGATE CHECK: At least one augmentation with >= 75% pass rate?")
    print(f"  Result: {'GO' if gate_pass else 'NO-GO'}")
    if best_strategy:
        print(f"  Best strategy: {best_strategy} (mean W1={best_w1:.1f})")

    output_data["gate"] = {
        "passed": gate_pass,
        "best_strategy": best_strategy,
        "best_w1": float(best_w1) if best_w1 != float("inf") else None,
    }
    output_data["total_elapsed_seconds"] = total_elapsed

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
