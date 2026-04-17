"""Phase 6c-1: H2 (features) + Power-boosted 6c (real Cora).

Phase 6f overturned Phase 6e's NO-GO by increasing seeds from 5 to 10. Phase 6c
(21 Cora subgraphs, p=0.748) may have been underpowered in the same way. This
experiment answers two questions simultaneously:

Part A (H2 — community features on Cora topology):
    Do shaping effects transfer to Cora's GRAPH topology when we replace real
    PCA-reduced features with SYNTHETIC community-boundary features? Uses Phase
    2g independent-sample protocol (100 train, 50 ref; independent seeds).

Part B (power-boosted 6c — real Cora features):
    Replicate Phase 6c's design (spectral augmentation from a single real
    feature matrix, reduced to d=16 by TruncatedSVD) but with 30 subgraphs
    instead of ~21. Same noise weights computed from training split only.

Outcome grid:
    H2 GO + Power-6c GO  -> shaping is real across settings, 6c was underpowered
    H2 GO + Power-6c NO  -> Cora's real features are the problem
    H2 NO + Power-6c GO  -> topology matters, Cora-specific failure (unlikely)
    H2 NO + Power-6c NO  -> Phase 2g effect is narrow; specaug (Framing B) primary
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
from graph_fans.phase6.augmentation import spectral_augmentation
from graph_fans.phase6.feature_reducer import fit_feature_reducer, reduce_features
from graph_fans.phase6.subgraph_sampler import sample_multiple_subgraphs
from graph_fans.utils.graph_generators import load_citation_network
from graph_fans.utils.multiscale_features import generate_community_boundary_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Shared config
N_NODES = 100
N_EPOCHS_DEFAULT = 500
N_TRAIN = 100
N_REF = 50
N_GEN = 50
B = 8

# Part A (H2 synthetic features on Cora topology)
N_DIMS_H2 = 4

# Part B (power-boosted 6c with real features)
N_DIMS_POWER = 16
SPECAUG_SIGMA = 0.3


# ---------------------------------------------------------------------------
# Part A: H2 — synthetic community features on Cora topology
# ---------------------------------------------------------------------------


def generate_independent_community_set(
    graph,
    n_samples: int,
    n_features: int,
    base_seed: int,
) -> np.ndarray:
    """Generate n_samples independent community-boundary feature realisations."""
    samples = []
    for i in range(n_samples):
        feat = generate_community_boundary_features(
            graph, n_features=n_features, seed=base_seed + i,
        )
        samples.append(feat)
    return np.stack(samples).astype(np.float32)


def run_subgraph_part_a(
    sample,
    n_epochs: int,
    device: str,
    subgraph_idx: int,
) -> dict:
    """Run Part A (H2) for one subgraph: indep community features, uniform vs band."""
    t0 = time.time()
    graph = sample.graph

    evals, evecs = compute_laplacian_spectrum(graph)
    _, bands = partition_into_bands(evals, B=B)

    # Independent community-boundary features on Cora subgraph topology
    # Seed offsets mirror Phase 6f so within-seed uniform/band share structure.
    train_data = generate_independent_community_set(
        graph,
        n_samples=N_TRAIN,
        n_features=N_DIMS_H2,
        base_seed=subgraph_idx * 10000,
    )
    ref_data = generate_independent_community_set(
        graph,
        n_samples=N_REF,
        n_features=N_DIMS_H2,
        base_seed=subgraph_idx * 10000 + 100000,
    )

    # Importance weights from first 50 training samples only (no ref leakage)
    profiles = []
    for feat in train_data[: min(50, len(train_data))]:
        profiles.append(compute_band_energy(feat, evecs, bands))
    weights = compute_importance_weights(np.mean(profiles, axis=0))

    # Spectral energy sanity
    band_energy = compute_band_energy(train_data[0], evecs, bands)
    energy_ratio = compute_energy_ratio(band_energy)

    results = {}
    for method, use_spectral in [("uniform", False), ("band", True)]:
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
            n_train_samples=N_TRAIN,
            use_spectral_noise=use_spectral,
            B=B,
        )
        trainer = Trainer(
            cfg, graph, train_data,
            importance_weights=weights if use_spectral else None,
        )
        history = trainer.train()
        gen = trainer.generate(n_samples=N_GEN)
        sanity = trainer.sanity_check(train_data, n_gen=5)
        w1 = spectral_w1_summary(ref_data, gen, evecs, bands)

        results[method] = {
            "final_loss": float(history["loss"][-1]),
            "std_ratio": float(sanity["std_ratio"]),
            "spectral_l2": float(sanity["spectral_l2"]),
            "w1_total": float(w1["total_w1"]),
            "w1_low": float(w1["low_band_w1"]),
            "w1_high": float(w1["high_band_w1"]),
            "per_band_w1": w1["per_band_w1"].tolist(),
        }

        logger.info(
            f"  [A sub={subgraph_idx} {method}] loss={history['loss'][-1]:.3f} "
            f"std_ratio={sanity['std_ratio']:.2f} W1={w1['total_w1']:.1f}"
        )

    delta_w1 = results["band"]["w1_total"] - results["uniform"]["w1_total"]
    elapsed = time.time() - t0
    logger.info(
        f"  [A sub={subgraph_idx}] delta_W1={delta_w1:+.1f} "
        f"({'band better' if delta_w1 < 0 else 'uniform better'}) "
        f"energy_ratio={energy_ratio:.1f} ({elapsed:.1f}s)"
    )

    return {
        "subgraph_idx": subgraph_idx,
        "uniform": results["uniform"],
        "band": results["band"],
        "delta_w1": float(delta_w1),
        "energy_ratio": float(energy_ratio),
        "elapsed_seconds": float(elapsed),
    }


# ---------------------------------------------------------------------------
# Part B: power-boosted 6c — real Cora features + spectral augmentation
# ---------------------------------------------------------------------------


def run_subgraph_part_b(
    sample,
    full_features: np.ndarray,
    reducer,
    n_epochs: int,
    device: str,
    subgraph_idx: int,
) -> dict:
    """Run Part B (powered 6c) for one subgraph: real features + specaug."""
    t0 = time.time()
    graph = sample.graph

    sub_features_full = full_features[sample.node_indices]
    sub_features = reduce_features(sub_features_full, reducer)

    evals, evecs = compute_laplacian_spectrum(graph)
    _, bands = partition_into_bands(evals, B=B)

    # Per Phase 6c pattern: spectral augmentation from the single real feature
    # matrix for both training and reference (but each uses its own aug seed,
    # following Phase 6f's logic for independent train/ref augmentations).
    train_data = spectral_augmentation(
        sub_features, evecs, n_samples=N_TRAIN,
        sigma=SPECAUG_SIGMA, seed=subgraph_idx,
    ).astype(np.float32)

    ref_data = spectral_augmentation(
        sub_features, evecs, n_samples=N_REF,
        sigma=SPECAUG_SIGMA, seed=subgraph_idx + 100000,
    ).astype(np.float32)

    # Importance weights from first 50 training samples only (no ref leakage)
    profiles = []
    for feat in train_data[: min(50, len(train_data))]:
        profiles.append(compute_band_energy(feat, evecs, bands))
    weights = compute_importance_weights(np.mean(profiles, axis=0))

    band_energy = compute_band_energy(sub_features, evecs, bands)
    energy_ratio = compute_energy_ratio(band_energy)

    results = {}
    for method, use_spectral in [("uniform", False), ("band", True)]:
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
            n_train_samples=N_TRAIN,
            use_spectral_noise=use_spectral,
            B=B,
        )
        trainer = Trainer(
            cfg, graph, train_data,
            importance_weights=weights if use_spectral else None,
        )
        history = trainer.train()
        gen = trainer.generate(n_samples=N_GEN)
        sanity = trainer.sanity_check(train_data, n_gen=5)
        w1 = spectral_w1_summary(ref_data, gen, evecs, bands)

        results[method] = {
            "final_loss": float(history["loss"][-1]),
            "std_ratio": float(sanity["std_ratio"]),
            "spectral_l2": float(sanity["spectral_l2"]),
            "w1_total": float(w1["total_w1"]),
            "w1_low": float(w1["low_band_w1"]),
            "w1_high": float(w1["high_band_w1"]),
            "per_band_w1": w1["per_band_w1"].tolist(),
        }

        logger.info(
            f"  [B sub={subgraph_idx} {method}] loss={history['loss'][-1]:.3f} "
            f"std_ratio={sanity['std_ratio']:.2f} W1={w1['total_w1']:.1f}"
        )

    delta_w1 = results["band"]["w1_total"] - results["uniform"]["w1_total"]
    elapsed = time.time() - t0
    logger.info(
        f"  [B sub={subgraph_idx}] delta_W1={delta_w1:+.1f} "
        f"({'band better' if delta_w1 < 0 else 'uniform better'}) "
        f"energy_ratio={energy_ratio:.1f} ({elapsed:.1f}s)"
    )

    return {
        "subgraph_idx": subgraph_idx,
        "uniform": results["uniform"],
        "band": results["band"],
        "delta_w1": float(delta_w1),
        "energy_ratio": float(energy_ratio),
        "elapsed_seconds": float(elapsed),
    }


# ---------------------------------------------------------------------------
# Summary / gate logic
# ---------------------------------------------------------------------------


def summarize(per_subgraph: list[dict], label: str) -> dict:
    """Compute paired t-test on (band - uniform) across subgraphs."""
    uniform_w1s = np.array([r["uniform"]["w1_total"] for r in per_subgraph])
    band_w1s = np.array([r["band"]["w1_total"] for r in per_subgraph])
    deltas = band_w1s - uniform_w1s  # negative = band better

    if len(uniform_w1s) >= 2:
        t_stat, p_two = stats.ttest_rel(band_w1s, uniform_w1s)
        # One-sided p-value for the alternative "band < uniform"
        p_one_sided = p_two / 2 if t_stat < 0 else 1 - p_two / 2
    else:
        t_stat, p_one_sided = 0.0, 1.0

    mean_uniform = float(uniform_w1s.mean())
    mean_band = float(band_w1s.mean())
    mean_delta = float(deltas.mean())
    improvement_pct = (
        -mean_delta / mean_uniform * 100.0 if mean_uniform > 0 else 0.0
    )
    n_band_better = int((deltas < 0).sum())

    gate_pass = bool(p_one_sided < 0.05 and mean_delta < 0)

    return {
        "label": label,
        "n_subgraphs": int(len(per_subgraph)),
        "mean_uniform_w1": mean_uniform,
        "std_uniform_w1": float(uniform_w1s.std()),
        "mean_band_w1": mean_band,
        "std_band_w1": float(band_w1s.std()),
        "mean_delta": mean_delta,
        "improvement_pct": float(improvement_pct),
        "n_band_better": n_band_better,
        "t_statistic": float(t_stat),
        "p_value_one_sided": float(p_one_sided),
        "gate_pass": gate_pass,
    }


def interpret(part_a_pass: bool, part_b_pass: bool) -> str:
    """Map the 2x2 outcome to a paper framing recommendation."""
    if part_a_pass and part_b_pass:
        return (
            "H2=GO, Power-6c=GO. Shaping is real across synthetic community features on Cora "
            "topology AND real Cora features (with power boost). Phase 6c was underpowered. "
            "Paper framing A (spectral shaping works) is viable."
        )
    if part_a_pass and not part_b_pass:
        return (
            "H2=GO, Power-6c=NO-GO. Shaping works on synthetic community features over Cora "
            "topology but not on real PCA-reduced Cora features. The Cora failure is feature-"
            "specific: real PCA'd features lack bimodal spectral structure. Paper framing B "
            "(spectral augmentation is the practical contribution) is primary, with shaping "
            "presented as conditional on feature structure."
        )
    if not part_a_pass and part_b_pass:
        return (
            "H2=NO-GO, Power-6c=GO. Contradictory and unexpected — shaping fails on synthetic "
            "community features on Cora topology yet succeeds on real Cora features. Likely "
            "indicates noise in one of the experiments or an unforeseen interaction with Cora "
            "topology. Needs follow-up."
        )
    return (
        "H2=NO-GO, Power-6c=NO-GO. Phase 2g's effect does not generalise to Cora's topology "
        "or features. Phase 2g's result is narrow (specific to synthetic SBM topology). "
        "Paper framing B (spectral augmentation from Phase 6b, 46% W1 improvement) should be "
        "primary. Noise shaping does not transfer to real citation networks."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6c-1: H2 + power-boosted 6c on Cora"
    )
    parser.add_argument("--n-subgraphs-h2", type=int, default=10)
    parser.add_argument("--n-subgraphs-power", type=int, default=30)
    parser.add_argument("--n-epochs", type=int, default=N_EPOCHS_DEFAULT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output",
        type=str,
        default="results/phase6c1/",
    )
    parser.add_argument("--quick", action="store_true", help="2 subgraphs each, 50 epochs")
    args = parser.parse_args()

    if args.quick:
        args.n_subgraphs_h2 = 2
        args.n_subgraphs_power = 2
        args.n_epochs = 50

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "h2_plus_power.json"

    logger.info("Phase 6c-1: H2 + power-boosted 6c")
    logger.info(
        f"  Part A (H2): {args.n_subgraphs_h2} Cora subgraphs, synthetic community features, d={N_DIMS_H2}"
    )
    logger.info(
        f"  Part B (powered 6c): {args.n_subgraphs_power} Cora subgraphs, real features PCA d={N_DIMS_POWER} + specaug"
    )
    logger.info(f"  n={N_NODES}, epochs={args.n_epochs}, device={args.device}")

    logger.info("Loading Cora citation network...")
    cora = load_citation_network("Cora")
    full_features = cora.features.astype(np.float64)
    logger.info(
        f"  Cora: {cora.graph.number_of_nodes()} nodes, "
        f"{cora.graph.number_of_edges()} edges, "
        f"features shape {full_features.shape}"
    )

    # Part A: BFS-sample subgraphs (use base_seed=0 so we share subgraph indices
    # with Phase 6c's initial subgraphs).
    logger.info(f"Sampling {args.n_subgraphs_h2} subgraphs for Part A (n={N_NODES})...")
    subgraphs_a = sample_multiple_subgraphs(
        cora.graph, N_NODES, k=args.n_subgraphs_h2, base_seed=0,
    )

    # Part B: BFS-sample subgraphs separately so we can extend beyond the A set
    # without collision. Use base_seed=1000 so we do not re-use A's subgraphs.
    logger.info(
        f"Sampling {args.n_subgraphs_power} subgraphs for Part B (n={N_NODES})..."
    )
    subgraphs_b = sample_multiple_subgraphs(
        cora.graph, N_NODES, k=args.n_subgraphs_power, base_seed=1000,
    )

    # Fit PCA once on full Cora for Part B
    logger.info(f"Fitting TruncatedSVD to d={N_DIMS_POWER} for Part B...")
    reducer = fit_feature_reducer(
        full_features, n_components=N_DIMS_POWER, method="truncated_svd", seed=0,
    )
    variance_explained = float(reducer.explained_variance_ratio_.sum())
    logger.info(f"  d={N_DIMS_POWER}: {variance_explained:.1%} variance explained")

    total_start = time.time()

    # -------------------------- Part A --------------------------
    logger.info("\n=== Part A (H2): synthetic community features on Cora topology ===")
    part_a_results: list[dict] = []
    for i, sample in enumerate(subgraphs_a):
        logger.info(
            f"\nPart A subgraph {i+1}/{args.n_subgraphs_h2} "
            f"(n={sample.graph.number_of_nodes()}, edges={sample.graph.number_of_edges()})"
        )
        result = run_subgraph_part_a(
            sample=sample,
            n_epochs=args.n_epochs,
            device=args.device,
            subgraph_idx=i,
        )
        part_a_results.append(result)

        # Intermediate save
        with open(out_path, "w") as f:
            json.dump(
                {
                    "status": "running_part_a",
                    "part_a": {"per_subgraph": part_a_results},
                },
                f,
                indent=2,
            )

    # -------------------------- Part B --------------------------
    logger.info(
        "\n=== Part B (powered 6c): real Cora features + spectral augmentation ==="
    )
    part_b_results: list[dict] = []
    for i, sample in enumerate(subgraphs_b):
        logger.info(
            f"\nPart B subgraph {i+1}/{args.n_subgraphs_power} "
            f"(n={sample.graph.number_of_nodes()}, edges={sample.graph.number_of_edges()})"
        )
        result = run_subgraph_part_b(
            sample=sample,
            full_features=full_features,
            reducer=reducer,
            n_epochs=args.n_epochs,
            device=args.device,
            subgraph_idx=i,
        )
        part_b_results.append(result)

        with open(out_path, "w") as f:
            json.dump(
                {
                    "status": "running_part_b",
                    "part_a": {"per_subgraph": part_a_results},
                    "part_b": {"per_subgraph": part_b_results},
                },
                f,
                indent=2,
            )

    total_elapsed = time.time() - total_start

    # -------------------------- Summaries --------------------------
    summary_a = summarize(part_a_results, label="Part A (H2: synthetic community features)")
    summary_b = summarize(part_b_results, label="Part B (power-boosted 6c: real features + specaug)")
    interpretation = interpret(summary_a["gate_pass"], summary_b["gate_pass"])

    logger.info("")
    logger.info("=" * 90)
    logger.info(f"Phase 6c-1 SUMMARY ({total_elapsed:.0f}s total)")
    logger.info("=" * 90)
    for name, summ in [("Part A (H2)", summary_a), ("Part B (Power-6c)", summary_b)]:
        logger.info(
            f"  {name}: uniform={summ['mean_uniform_w1']:.1f}  "
            f"band={summ['mean_band_w1']:.1f}  "
            f"improv={summ['improvement_pct']:+.1f}%  "
            f"p={summ['p_value_one_sided']:.4f}  "
            f"{'GO' if summ['gate_pass'] else 'NO-GO'}"
        )
    logger.info("")
    logger.info(f"Interpretation: {interpretation}")

    output_data = {
        "experiment": "phase6c1_h2_plus_power",
        "dataset": "Cora",
        "n_nodes": N_NODES,
        "n_epochs": args.n_epochs,
        "part_a": {
            "protocol": "synthetic community-boundary features on Cora topology, indep train/ref",
            "n_subgraphs": args.n_subgraphs_h2,
            "n_dims": N_DIMS_H2,
            "n_train": N_TRAIN,
            "n_ref": N_REF,
            "n_gen": N_GEN,
            "per_subgraph": part_a_results,
            "summary": summary_a,
            "gate_pass": summary_a["gate_pass"],
        },
        "part_b": {
            "protocol": "real Cora features, TruncatedSVD d=16, spectral augmentation sigma=0.3",
            "n_subgraphs": args.n_subgraphs_power,
            "n_dims": N_DIMS_POWER,
            "variance_explained": variance_explained,
            "specaug_sigma": SPECAUG_SIGMA,
            "n_train": N_TRAIN,
            "n_ref": N_REF,
            "n_gen": N_GEN,
            "per_subgraph": part_b_results,
            "summary": summary_b,
            "gate_pass": summary_b["gate_pass"],
        },
        "interpretation": interpretation,
        "combined_gate_grid": {
            "part_a_pass": summary_a["gate_pass"],
            "part_b_pass": summary_b["gate_pass"],
        },
        "total_elapsed_seconds": total_elapsed,
        "status": "complete",
    }

    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
