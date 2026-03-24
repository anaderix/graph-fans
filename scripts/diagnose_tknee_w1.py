#!/usr/bin/env python
"""Diagnostic: validate t_knee effect using spectral-domain W1 distance.

Trains lightweight models with different t_knee values on a single family/seed,
generates samples, and compares per-band W1 between generated and reference.

Usage:
    uv run python scripts/diagnose_tknee_w1.py \
        --dataset-dir results/phase2f_small/datasets \
        --family "SBM(q=0.01)" --seed 0 \
        --epochs 500 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from graph_fans.phase0.spectral_profiler import compute_laplacian_spectrum, partition_into_bands
from graph_fans.phase2.dataset import load_dataset
from graph_fans.phase2.evaluate import _get_graph
from graph_fans.phase2.noise_shaper import compute_importance_weights
from graph_fans.phase2.spectral_wasserstein import per_band_w1, spectral_w1_summary
from graph_fans.phase2.trainer import Trainer, TrainConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

N_GEN = 50  # samples to generate per run


def run_diagnostic(
    dataset_dir: str,
    family: str,
    seed: int,
    n_nodes: int,
    epochs: int,
    device: str,
    t_knee_values: list[float],
    B: int = 8,
    output_dir: str = "results/diagnostics",
):
    # Load cached dataset
    safe_family = family.replace("(", "_").replace(")", "").replace("=", "")
    ds_path = Path(dataset_dir) / f"{safe_family}_seed{seed}.npz"
    if not ds_path.exists():
        logger.error(f"Dataset not found: {ds_path}")
        return

    ds = load_dataset(ds_path)
    train_data = ds["train"]
    ref_data = ds["ref"]
    n_features = train_data.shape[2]

    logger.info(f"Loaded dataset: train={train_data.shape}, ref={ref_data.shape}")

    # Graph + spectral decomposition
    graph = _get_graph(family, n_nodes, seed=0)
    eigenvalues, eigenvectors = compute_laplacian_spectrum(graph)
    _, band_indices = partition_into_bands(eigenvalues, B=B)

    # Importance weights from training data spectral profile
    from graph_fans.phase0.spectral_profiler import compute_band_energy
    profiles = []
    for feat in train_data[:50]:
        e = compute_band_energy(feat, eigenvectors, band_indices)
        profiles.append(e)
    mean_energy = np.mean(profiles, axis=0)
    weights = compute_importance_weights(mean_energy)

    # Baseline: W1 between ref and random noise
    rng = np.random.RandomState(42)
    noise_samples = rng.randn(N_GEN, train_data.shape[1], n_features)
    noise_w1 = per_band_w1(ref_data, noise_samples, eigenvectors, band_indices)

    # Baseline: W1 between ref and a second ref draw (train data as proxy)
    oracle_w1 = per_band_w1(ref_data, train_data[:N_GEN], eigenvectors, band_indices)

    results = {
        "family": family,
        "seed": seed,
        "n_nodes": n_nodes,
        "epochs": epochs,
        "B": B,
        "noise_baseline_w1": noise_w1.tolist(),
        "oracle_baseline_w1": oracle_w1.tolist(),
        "runs": [],
    }

    # Run: uniform (no shaping) as control
    logger.info(f"\n=== uniform (no shaping) ===")
    cfg = TrainConfig(
        n_epochs=epochs, batch_timesteps=32, seed=seed, device=device,
        hidden_dim=128, n_layers=3, n_gen_steps=200,
        sde_type="cosine", use_ema=True, use_lr_scheduler=True,
        use_spectral_noise=False, n_train_samples=train_data.shape[0],
    )
    start = time.time()
    trainer = Trainer(cfg, graph, train_data, None)
    history = trainer.train()
    gen = trainer.generate(n_samples=N_GEN)
    elapsed = time.time() - start

    summary = spectral_w1_summary(ref_data, gen, eigenvectors, band_indices)
    results["runs"].append({
        "method": "uniform",
        "t_knee": None,
        "final_loss": history["loss"][-1],
        "gen_std": float(gen.std()),
        "train_time_s": elapsed,
        **{f"band{b}_w1": float(summary["per_band_w1"][b]) for b in range(B)},
        "total_w1": summary["total_w1"],
        "low_band_w1": summary["low_band_w1"],
        "high_band_w1": summary["high_band_w1"],
    })

    # Run: fully shaped (no ramp)
    logger.info(f"\n=== spectral (fully shaped, no ramp) ===")
    cfg.use_spectral_noise = True
    cfg.t_knee = None
    trainer = Trainer(cfg, graph, train_data, weights)
    history = trainer.train()
    gen = trainer.generate(n_samples=N_GEN)

    summary = spectral_w1_summary(ref_data, gen, eigenvectors, band_indices)
    results["runs"].append({
        "method": "spectral",
        "t_knee": None,
        "final_loss": history["loss"][-1],
        "gen_std": float(gen.std()),
        "train_time_s": elapsed,
        **{f"band{b}_w1": float(summary["per_band_w1"][b]) for b in range(B)},
        "total_w1": summary["total_w1"],
        "low_band_w1": summary["low_band_w1"],
        "high_band_w1": summary["high_band_w1"],
    })

    # Run: t_knee grid
    for t_knee in t_knee_values:
        logger.info(f"\n=== spectral_ramp t_knee={t_knee} ===")
        cfg.use_spectral_noise = True
        cfg.t_knee = t_knee
        start = time.time()
        trainer = Trainer(cfg, graph, train_data, weights)
        history = trainer.train()
        gen = trainer.generate(n_samples=N_GEN)
        elapsed = time.time() - start

        summary = spectral_w1_summary(ref_data, gen, eigenvectors, band_indices)
        results["runs"].append({
            "method": f"spectral_ramp",
            "t_knee": t_knee,
            "final_loss": history["loss"][-1],
            "gen_std": float(gen.std()),
            "train_time_s": elapsed,
            **{f"band{b}_w1": float(summary["per_band_w1"][b]) for b in range(B)},
            "total_w1": summary["total_w1"],
            "low_band_w1": summary["low_band_w1"],
            "high_band_w1": summary["high_band_w1"],
        })

    # Print summary table
    print("\n" + "=" * 120)
    print(f"SPECTRAL W1 DIAGNOSTIC: {family}, seed={seed}, {n_nodes} nodes, {epochs} epochs")
    print("=" * 120)

    header = f"{'Method':<25} {'t_knee':>6} {'Loss':>7} {'GenStd':>7} {'Total':>8}"
    for b in range(B):
        header += f" {'B'+str(b):>7}"
    print(header)
    print("-" * 120)

    # Baselines
    row = f"{'[oracle: train vs ref]':<25} {'—':>6} {'—':>7} {'—':>7} {oracle_w1.sum():>8.2f}"
    for b in range(B):
        row += f" {oracle_w1[b]:>7.2f}"
    print(row)

    row = f"{'[noise baseline]':<25} {'—':>6} {'—':>7} {'—':>7} {noise_w1.sum():>8.2f}"
    for b in range(B):
        row += f" {noise_w1[b]:>7.2f}"
    print(row)
    print("-" * 120)

    for run in results["runs"]:
        tk = f"{run['t_knee']:.2f}" if run["t_knee"] is not None else "—"
        row = f"{run['method']:<25} {tk:>6} {run['final_loss']:>7.3f} {run['gen_std']:>7.2f} {run['total_w1']:>8.2f}"
        for b in range(B):
            row += f" {run[f'band{b}_w1']:>7.2f}"
        print(row)

    print("-" * 120)
    print("Lower W1 = better. Oracle = irreducible sampling noise. Noise = worst case.")

    # Save JSON
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / f"tknee_w1_{safe_family}_seed{seed}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {json_path}")


def main():
    parser = argparse.ArgumentParser(description="t_knee diagnostic via spectral W1")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--family", default="SBM(q=0.01)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-nodes", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--t-knee-values", default="0.05,0.10,0.15,0.20,0.30")
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--output-dir", default="results/diagnostics")
    args = parser.parse_args()

    t_knees = [float(x) for x in args.t_knee_values.split(",")]

    run_diagnostic(
        dataset_dir=args.dataset_dir,
        family=args.family,
        seed=args.seed,
        n_nodes=args.n_nodes,
        epochs=args.epochs,
        device=args.device,
        t_knee_values=t_knees,
        B=args.bands,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
