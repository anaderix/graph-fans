"""Phase 6a: Scale diagnostic for real Cora citation network features on subgraphs.

Tests whether the 3L GCN can denoise PCA-reduced Cora features on n=50-200
subgraphs. For each (n, d) combination, samples BFS subgraphs from Cora,
reduces features via TruncatedSVD (fit on full Cora), creates Gaussian-augmented
training data, trains baseline diffusion, and evaluates with W1 + sanity checks.

Gate: GCN denoises at n=100 d=16 (std_ratio < 3.0 for >=15/20 subgraphs)
AND spectral energy profile is non-uniform (energy ratio >= 2x).
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
from graph_fans.utils.graph_generators import load_citation_network

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

N_GEN = 50  # Number of generated samples for W1 evaluation
N_AUG = 100  # Number of augmented training samples


def run_scale_cell(
    full_graph_data,
    full_features: np.ndarray,
    n_nodes: int,
    n_dims: int,
    n_subgraphs: int,
    n_epochs: int,
    device: str,
    reducer,
) -> dict:
    """Run evaluation for one (n_nodes, n_dims) cell.

    Args:
        full_graph_data: GraphData from load_citation_network.
        full_features: Full Cora feature matrix [2708, 1433].
        n_nodes: Subgraph size.
        n_dims: Feature dimensionality after reduction.
        n_subgraphs: Number of BFS subgraphs to sample.
        n_epochs: Training epochs.
        device: Torch device.
        reducer: Pre-fitted TruncatedSVD for this n_dims.

    Returns:
        Dict with per-subgraph results and aggregate statistics.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"  n={n_nodes}, d={n_dims}: sampling {n_subgraphs} subgraphs")
    logger.info(f"{'='*70}")

    subgraphs = sample_multiple_subgraphs(
        full_graph_data.graph, n_nodes, k=n_subgraphs, base_seed=0
    )

    per_subgraph: list[dict] = []
    std_ratios: list[float] = []
    spectral_l2s: list[float] = []
    w1_totals: list[float] = []
    energy_ratios: list[float] = []
    pass_count = 0

    for i, sample in enumerate(subgraphs):
        t0 = time.time()
        logger.info(f"\n  Subgraph {i+1}/{n_subgraphs} (n={sample.graph.number_of_nodes()} nodes)")

        # Extract and reduce features for this subgraph's nodes
        sub_features_full = full_features[sample.node_indices]  # [n, 1433]
        sub_features = reduce_features(sub_features_full, reducer)  # [n, d]

        # Spectral energy profile of reduced features
        evals, evecs = compute_laplacian_spectrum(sample.graph)
        _, bands = partition_into_bands(evals, B=8)
        band_energy = compute_band_energy(sub_features, evecs, bands)
        energy_ratio = compute_energy_ratio(band_energy)
        total_energy = band_energy.sum()
        norm_profile = (band_energy / total_energy).tolist() if total_energy > 0 else band_energy.tolist()

        energy_ratios.append(energy_ratio)
        logger.info(f"    Energy ratio: {energy_ratio:.2f}")

        # Create augmented training data
        train_data = gaussian_augmentation(
            sub_features, n_samples=N_AUG, seed=i
        )

        # Reference: the un-augmented real features (single sample, expanded)
        ref_data = sub_features[np.newaxis, ...]  # [1, n, d]

        # Train baseline (uniform noise)
        cfg = TrainConfig(
            n_epochs=n_epochs,
            batch_timesteps=32,
            seed=i,
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

        # Sanity check
        sanity = trainer.sanity_check(train_data, n_gen=5)
        std_ratio = sanity["std_ratio"]
        spectral_l2 = sanity["spectral_l2"]
        std_ratios.append(std_ratio)
        spectral_l2s.append(spectral_l2)

        passed = std_ratio < 3.0
        if passed:
            pass_count += 1

        # Generate samples and compute W1 vs real features
        gen_data = trainer.generate(n_samples=N_GEN)
        w1 = spectral_w1_summary(ref_data, gen_data, evecs, bands)
        w1_totals.append(float(w1["total_w1"]))

        elapsed = time.time() - t0
        logger.info(
            f"    loss={history['loss'][-1]:.4f}, "
            f"std_ratio={std_ratio:.2f}, "
            f"spectral_L2={spectral_l2:.3f}, "
            f"W1={w1['total_w1']:.1f}, "
            f"{'PASS' if passed else 'FAIL'} "
            f"({elapsed:.1f}s)"
        )

        per_subgraph.append({
            "subgraph_idx": i,
            "n_actual_nodes": sample.graph.number_of_nodes(),
            "final_loss": float(history["loss"][-1]),
            "std_ratio": std_ratio,
            "spectral_l2": spectral_l2,
            "w1_total": float(w1["total_w1"]),
            "w1_low": float(w1["low_band_w1"]),
            "w1_high": float(w1["high_band_w1"]),
            "per_band_w1": w1["per_band_w1"].tolist(),
            "energy_ratio": energy_ratio,
            "spectral_profile": norm_profile,
            "passed": passed,
            "warnings": sanity["warnings"],
        })

    result = {
        "n_nodes": n_nodes,
        "n_dims": n_dims,
        "n_subgraphs": n_subgraphs,
        "n_epochs": n_epochs,
        "pass_count": pass_count,
        "pass_rate": pass_count / max(n_subgraphs, 1),
        "std_ratio_mean": float(np.mean(std_ratios)),
        "std_ratio_median": float(np.median(std_ratios)),
        "spectral_l2_mean": float(np.mean(spectral_l2s)),
        "w1_total_mean": float(np.mean(w1_totals)),
        "w1_total_std": float(np.std(w1_totals)),
        "energy_ratio_mean": float(np.mean(energy_ratios)),
        "energy_ratio_median": float(np.median(energy_ratios)),
        "per_subgraph": per_subgraph,
    }

    logger.info(f"\n  n={n_nodes}, d={n_dims} SUMMARY:")
    logger.info(f"    Pass rate: {pass_count}/{n_subgraphs} ({result['pass_rate']:.0%})")
    logger.info(f"    std_ratio: mean={result['std_ratio_mean']:.2f}, median={result['std_ratio_median']:.2f}")
    logger.info(f"    spectral_L2: mean={result['spectral_l2_mean']:.3f}")
    logger.info(f"    W1: {result['w1_total_mean']:.1f} +/- {result['w1_total_std']:.1f}")
    logger.info(f"    Energy ratio: mean={result['energy_ratio_mean']:.2f}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6a: Scale diagnostic for Cora subgraph denoising"
    )
    parser.add_argument(
        "--n-scales", type=str, default="50,100,150,200",
        help="Comma-separated subgraph sizes (default: 50,100,150,200)",
    )
    parser.add_argument(
        "--n-dims", type=str, default="4,16,32",
        help="Comma-separated feature dimensions after PCA (default: 4,16,32)",
    )
    parser.add_argument(
        "--n-subgraphs", type=int, default=20,
        help="Number of BFS subgraphs per (n, d) cell (default: 20)",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=500,
        help="Training epochs per run (default: 500)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device: 'cuda' or 'cpu' (default: cuda)",
    )
    parser.add_argument(
        "--output", type=str, default="results/phase6a/scale_diagnostic.json",
        help="Output JSON path (default: results/phase6a/scale_diagnostic.json)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 3 subgraphs, 50 epochs (for testing)",
    )
    args = parser.parse_args()

    scales = [int(s.strip()) for s in args.n_scales.split(",")]
    dims = [int(d.strip()) for d in args.n_dims.split(",")]

    n_subgraphs = 3 if args.quick else args.n_subgraphs
    n_epochs = 50 if args.quick else args.n_epochs

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Load Cora
    logger.info("Loading Cora citation network...")
    cora = load_citation_network("Cora")
    full_features = cora.features.astype(np.float64)
    logger.info(
        f"  Cora: {cora.graph.number_of_nodes()} nodes, "
        f"{cora.graph.number_of_edges()} edges, "
        f"features shape {full_features.shape}"
    )

    # Step 2: Fit TruncatedSVD for each d (on FULL Cora, no data leakage)
    reducers = {}
    for d in dims:
        logger.info(f"Fitting TruncatedSVD to d={d}...")
        reducer = fit_feature_reducer(full_features, n_components=d, method="truncated_svd", seed=0)
        variance_explained = reducer.explained_variance_ratio_.sum()
        logger.info(f"  d={d}: {variance_explained:.1%} variance explained")
        reducers[d] = reducer

    # Step 3: Run each (n, d) cell
    all_results: list[dict] = []
    total_start = time.time()

    for n in scales:
        for d in dims:
            cell_result = run_scale_cell(
                full_graph_data=cora,
                full_features=full_features,
                n_nodes=n,
                n_dims=d,
                n_subgraphs=n_subgraphs,
                n_epochs=n_epochs,
                device=args.device,
                reducer=reducers[d],
            )
            all_results.append(cell_result)

            # Save after each cell (crash-safe)
            output_data = {
                "experiment": "phase6a_scale_diagnostic",
                "dataset": "Cora",
                "n_full_nodes": cora.graph.number_of_nodes(),
                "feature_dim_original": full_features.shape[1],
                "reducers": {
                    str(d2): {
                        "method": "truncated_svd",
                        "variance_explained": float(reducers[d2].explained_variance_ratio_.sum()),
                    }
                    for d2 in dims
                },
                "results": all_results,
            }
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"  Saved intermediate results to {output_path}")

    total_elapsed = time.time() - total_start

    # Step 4: Summary table
    print(f"\n{'='*90}")
    print(f"Phase 6a Scale Diagnostic — Cora ({total_elapsed:.0f}s total)")
    print(f"{'='*90}")
    print(
        f"{'n':>5} {'d':>4} {'Pass':>6} {'Rate':>6} "
        f"{'std_r':>7} {'specL2':>7} {'W1':>8} {'E_ratio':>8}"
    )
    print("-" * 90)

    gate_n100_d16 = None
    for r in all_results:
        rate_str = f"{r['pass_count']}/{r['n_subgraphs']}"
        print(
            f"{r['n_nodes']:>5} {r['n_dims']:>4} {rate_str:>6} "
            f"{r['pass_rate']:>5.0%} "
            f"{r['std_ratio_mean']:>7.2f} "
            f"{r['spectral_l2_mean']:>7.3f} "
            f"{r['w1_total_mean']:>7.1f}+/-{r['w1_total_std']:<4.1f} "
            f"{r['energy_ratio_mean']:>7.1f}"
        )
        if r["n_nodes"] == 100 and r["n_dims"] == 16:
            gate_n100_d16 = r

    print(f"{'='*90}")

    # Gate check
    if gate_n100_d16 is not None:
        gate_pass_rate = gate_n100_d16["pass_rate"]
        gate_energy = gate_n100_d16["energy_ratio_mean"]
        pass_threshold = 15 / 20  # 75%
        energy_threshold = 2.0

        denoise_ok = gate_pass_rate >= pass_threshold
        spectral_ok = gate_energy >= energy_threshold

        print(f"\nGATE CHECK (n=100, d=16):")
        print(f"  Denoising: {gate_n100_d16['pass_count']}/{gate_n100_d16['n_subgraphs']} "
              f"pass (need >= 75%) — {'GO' if denoise_ok else 'NO-GO'}")
        print(f"  Energy ratio: {gate_energy:.1f}x (need >= 2.0x) — {'GO' if spectral_ok else 'NO-GO'}")
        print(f"  OVERALL: {'GO' if (denoise_ok and spectral_ok) else 'NO-GO'}")

        # Save gate result
        output_data["gate"] = {
            "cell": "n=100,d=16",
            "pass_rate": gate_pass_rate,
            "pass_threshold": pass_threshold,
            "denoise_ok": denoise_ok,
            "energy_ratio": gate_energy,
            "energy_threshold": energy_threshold,
            "spectral_ok": spectral_ok,
            "overall": denoise_ok and spectral_ok,
        }
    else:
        print("\nWARNING: n=100,d=16 cell not found — gate check skipped")
        print("  (Run with --n-scales including 100 and --n-dims including 16)")

    # Final save
    output_data["total_elapsed_seconds"] = total_elapsed
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
