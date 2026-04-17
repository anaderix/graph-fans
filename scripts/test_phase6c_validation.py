"""Phase 6c: Core shaping validation on real Cora features.

Tests whether band-shaped noise improves W1 vs uniform noise on Cora subgraphs
using the best augmentation strategy from Phase 6b.

For each of 50 BFS subgraphs:
  1. Extract and reduce features (TruncatedSVD d=16, fit on full Cora)
  2. Create augmented training data using best augmentation from 6b
  3. Train uniform vs band-shaped diffusion
  4. Generate samples and evaluate W1 vs un-augmented real features
  5. Paired t-test across 50 subgraphs

Gate: Band-shaped W1 < uniform W1 with p < 0.05.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from scipy import stats

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


def run_pair(
    sample,
    sub_features: np.ndarray,
    evecs: np.ndarray,
    bands: list[np.ndarray],
    augmentation: str,
    n_epochs: int,
    device: str,
    subgraph_idx: int,
) -> dict:
    """Run uniform vs band-shaped for one subgraph."""
    t0 = time.time()

    train_data = create_augmented_data(
        augmentation, sub_features, evecs, n_samples=N_AUG, seed=subgraph_idx,
    )

    mean_band_energy = np.zeros(len(bands))
    for s in range(min(train_data.shape[0], 50)):
        mean_band_energy += compute_band_energy(train_data[s], evecs, bands)
    mean_band_energy /= min(train_data.shape[0], 50)
    importance_weights = compute_importance_weights(mean_band_energy)

    ref_data = sub_features[np.newaxis, ...]
    results = {}

    for noise_method in ["uniform", "band"]:
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
        iw = importance_weights if noise_method == "band" else None
        trainer = Trainer(cfg, sample.graph, train_data, importance_weights=iw)
        history = trainer.train()

        sanity = trainer.sanity_check(train_data, n_gen=5)
        gen_data = trainer.generate(n_samples=N_GEN)
        w1 = spectral_w1_summary(ref_data, gen_data, evecs, bands)

        results[noise_method] = {
            "final_loss": float(history["loss"][-1]),
            "std_ratio": sanity["std_ratio"],
            "spectral_l2": sanity["spectral_l2"],
            "w1_total": float(w1["total_w1"]),
            "w1_low": float(w1["low_band_w1"]),
            "w1_high": float(w1["high_band_w1"]),
            "per_band_w1": w1["per_band_w1"].tolist(),
            "passed": sanity["std_ratio"] < 3.0,
        }

        logger.info(
            f"    {noise_method}: "
            f"loss={history['loss'][-1]:.4f}, "
            f"std_ratio={sanity['std_ratio']:.2f}, "
            f"W1={w1['total_w1']:.1f}"
        )

    elapsed = time.time() - t0
    delta_w1 = results["band"]["w1_total"] - results["uniform"]["w1_total"]
    logger.info(
        f"    delta_W1={delta_w1:+.1f} "
        f"({'band better' if delta_w1 < 0 else 'uniform better'}) "
        f"({elapsed:.1f}s)"
    )

    return {
        "subgraph_idx": subgraph_idx,
        "uniform": results["uniform"],
        "band": results["band"],
        "delta_w1": delta_w1,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6c: Core shaping validation on Cora subgraphs"
    )
    parser.add_argument(
        "--n-subgraphs", type=int, default=50,
        help="Number of BFS subgraphs (default: 50)",
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
        "--augmentation", type=str, default="gaussian",
        choices=["gaussian", "spectral", "dropout"],
        help="Augmentation strategy from Phase 6b (default: gaussian)",
    )
    parser.add_argument(
        "--output", type=str, default="results/phase6c/shaping_validation.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 5 subgraphs, 50 epochs",
    )
    args = parser.parse_args()

    n_subgraphs = 5 if args.quick else args.n_subgraphs
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

        result = run_pair(
            sample=sample,
            sub_features=sub_features,
            evecs=evecs,
            bands=bands,
            augmentation=args.augmentation,
            n_epochs=n_epochs,
            device=args.device,
            subgraph_idx=i,
        )
        result["energy_ratio"] = energy_ratio
        all_results.append(result)

        output_data = {
            "experiment": "phase6c_shaping_validation",
            "dataset": "Cora",
            "n_nodes": N_NODES,
            "n_dims": N_DIMS,
            "augmentation": args.augmentation,
            "n_subgraphs": n_subgraphs,
            "n_epochs": n_epochs,
            "variance_explained": float(variance_explained),
            "results": all_results,
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

    total_elapsed = time.time() - total_start

    # Paired t-test
    uniform_w1s = np.array([r["uniform"]["w1_total"] for r in all_results])
    band_w1s = np.array([r["band"]["w1_total"] for r in all_results])
    deltas = band_w1s - uniform_w1s

    t_stat, p_value_two = stats.ttest_rel(band_w1s, uniform_w1s)
    p_value = p_value_two / 2 if t_stat < 0 else 1 - p_value_two / 2

    n_band_better = (deltas < 0).sum()
    mean_delta = deltas.mean()
    mean_uniform = uniform_w1s.mean()
    mean_band = band_w1s.mean()
    improvement_pct = -mean_delta / mean_uniform * 100 if mean_uniform > 0 else 0

    print(f"\n{'='*90}")
    print(f"Phase 6c Shaping Validation ({total_elapsed:.0f}s total)")
    print(f"{'='*90}")
    print(f"  Augmentation: {args.augmentation}")
    print(f"  Subgraphs: {n_subgraphs}")
    print(f"  Uniform W1: {mean_uniform:.1f} +/- {uniform_w1s.std():.1f}")
    print(f"  Band W1:    {mean_band:.1f} +/- {band_w1s.std():.1f}")
    print(f"  Delta:      {mean_delta:+.1f} ({improvement_pct:+.1f}%)")
    print(f"  Band better: {n_band_better}/{n_subgraphs}")
    print(f"  Paired t-test: t={t_stat:.3f}, p={p_value:.4f} (one-sided)")

    gate_pass = p_value < 0.05 and mean_delta < 0
    print(f"\nGATE CHECK: Band-shaped W1 < uniform W1 with p < 0.05?")
    print(f"  Result: {'GO' if gate_pass else 'NO-GO'}")

    output_data["statistics"] = {
        "mean_uniform_w1": float(mean_uniform),
        "std_uniform_w1": float(uniform_w1s.std()),
        "mean_band_w1": float(mean_band),
        "std_band_w1": float(band_w1s.std()),
        "mean_delta": float(mean_delta),
        "improvement_pct": float(improvement_pct),
        "n_band_better": int(n_band_better),
        "t_statistic": float(t_stat),
        "p_value_one_sided": float(p_value),
    }
    output_data["gate"] = {
        "passed": gate_pass,
        "p_value": float(p_value),
        "mean_delta": float(mean_delta),
        "improvement_pct": float(improvement_pct),
    }
    output_data["total_elapsed_seconds"] = total_elapsed

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
