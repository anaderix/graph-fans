"""Phase 5a experiment: Independent Per-Mode Spectral Diffusion.

3-way comparison:
  1. uniform_spatial  -- Phase 2 baseline (GCN, uniform noise)
  2. band_spatial     -- Phase 2g best (GCN, band-shaped noise)
  3. spectral_5a      -- New: shared MLP in spectral domain, per-mode DDIM

All methods share the same dataset, graph, eigendecomposition, and W1 metric.
Paired t-test with Bonferroni correction (4 families, alpha=0.0125).

Data leakage safeguards:
- Importance weights from seed-0 training split only.
- ds["ref"] is never accessed during training or weight computation.
- Each seed uses an independently generated dataset split.
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
    compute_laplacian_spectrum,
    partition_into_bands,
)
from graph_fans.phase2.dataset import get_or_generate_dataset
from graph_fans.phase2.evaluate import _get_graph
from graph_fans.phase2.noise_shaper import compute_importance_weights
from graph_fans.phase2.spectral_wasserstein import spectral_w1_summary
from graph_fans.phase2.trainer import Trainer, TrainConfig
from graph_fans.phase5.spatial_coherence import spatial_coherence_summary
from graph_fans.phase5.spectral_trainer import SpectralTrainer, SpectralTrainConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

N_GEN = 50
DEFAULT_FAMILIES = "SBM(q=0.05),SBM(q=0.1),BA(m=2),BA(m=5)"


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
    energy_exponent: float = 0.5,
) -> dict:
    """Run 3-way comparison for one graph family.

    Returns dict with per-method W1 arrays and statistical test results.
    """
    logger.info("\n%s", "=" * 80)
    logger.info("  %s (n=%d)", family, n_nodes)
    logger.info("%s", "=" * 80)

    graph = _get_graph(family, n_nodes, seed=0)
    evals, evecs = compute_laplacian_spectrum(graph)
    _, bands = partition_into_bands(evals, B=8)

    # Compute importance weights from seed-0 TRAINING split only
    ds0 = get_or_generate_dataset(
        graph, family, 0,
        n_train=100, n_ref=50,
        n_features=4, feature_mode="community",
        cache_dir=dataset_dir,
    )
    profiles = []
    for feat in ds0["train"][:50]:
        e = compute_band_energy(feat, evecs, bands)
        profiles.append(e)
    weights = compute_importance_weights(np.mean(profiles, axis=0))
    logger.info("  Importance weights (from seed-0 train): %s", weights.weights.round(3))

    uniform_w1s: list[float] = []
    band_w1s: list[float] = []
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

        # --- Method 1: uniform_spatial (Phase 2 baseline) ---
        cfg_uniform = TrainConfig(
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
        )
        t0 = time.time()
        trainer_uniform = Trainer(cfg_uniform, graph, ds["train"])
        hist_uniform = trainer_uniform.train()
        gen_uniform = trainer_uniform.generate(n_samples=N_GEN)
        time_uniform = time.time() - t0

        w1_uniform = spectral_w1_summary(ds["ref"], gen_uniform, evecs, bands)
        seed_entry["uniform"] = {
            "w1_total": float(w1_uniform["total_w1"]),
            "w1_low": float(w1_uniform["low_band_w1"]),
            "w1_high": float(w1_uniform["high_band_w1"]),
            "per_band_w1": w1_uniform["per_band_w1"].tolist(),
            "final_loss": float(hist_uniform["loss"][-1]),
            "time_s": time_uniform,
        }
        uniform_w1s.append(float(w1_uniform["total_w1"]))
        logger.info(
            "  %s/seed=%d/uniform: loss=%.3f, W1=%.1f (low=%.1f, high=%.1f), time=%.0fs",
            family, seed, hist_uniform["loss"][-1],
            w1_uniform["total_w1"], w1_uniform["low_band_w1"], w1_uniform["high_band_w1"],
            time_uniform,
        )

        # --- Method 2: band_spatial (Phase 2g best) ---
        cfg_band = TrainConfig(
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
            noise_shaping="band",
        )
        t0 = time.time()
        trainer_band = Trainer(cfg_band, graph, ds["train"], importance_weights=weights)
        hist_band = trainer_band.train()
        gen_band = trainer_band.generate(n_samples=N_GEN)
        time_band = time.time() - t0

        w1_band = spectral_w1_summary(ds["ref"], gen_band, evecs, bands)
        seed_entry["band"] = {
            "w1_total": float(w1_band["total_w1"]),
            "w1_low": float(w1_band["low_band_w1"]),
            "w1_high": float(w1_band["high_band_w1"]),
            "per_band_w1": w1_band["per_band_w1"].tolist(),
            "final_loss": float(hist_band["loss"][-1]),
            "time_s": time_band,
        }
        band_w1s.append(float(w1_band["total_w1"]))
        logger.info(
            "  %s/seed=%d/band: loss=%.3f, W1=%.1f (low=%.1f, high=%.1f), time=%.0fs",
            family, seed, hist_band["loss"][-1],
            w1_band["total_w1"], w1_band["low_band_w1"], w1_band["high_band_w1"],
            time_band,
        )

        # --- Method 3: spectral_5a (new) ---
        spec_cfg = SpectralTrainConfig(
            n_epochs=n_epochs,
            lr=1e-3,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_features=4,
            energy_exponent=energy_exponent,
            seed=seed,
            device=device,
            use_ema=True,
            use_lr_scheduler=True,
        )
        t0 = time.time()
        spec_trainer = SpectralTrainer(graph, ds["train"], spec_cfg)
        hist_spectral = spec_trainer.train()
        gen_spectral = spec_trainer.generate(n_samples=N_GEN)
        time_spectral = time.time() - t0

        # Stack for W1 evaluation (W1 expects [N, n_nodes, n_features])
        gen_spectral_arr = np.stack(gen_spectral)
        w1_spectral = spectral_w1_summary(ds["ref"], gen_spectral_arr, evecs, bands)

        # Spatial coherence diagnostics
        coherence = spatial_coherence_summary(
            ds["ref"][0], gen_spectral[0], graph, evecs,
        )

        seed_entry["spectral"] = {
            "w1_total": float(w1_spectral["total_w1"]),
            "w1_low": float(w1_spectral["low_band_w1"]),
            "w1_high": float(w1_spectral["high_band_w1"]),
            "per_band_w1": w1_spectral["per_band_w1"].tolist(),
            "final_loss": float(hist_spectral["final_loss"]),
            "time_s": time_spectral,
            "coherence": coherence,
        }
        spectral_w1s.append(float(w1_spectral["total_w1"]))
        logger.info(
            "  %s/seed=%d/spectral: loss=%.3f, W1=%.1f (low=%.1f, high=%.1f), time=%.0fs",
            family, seed, hist_spectral["final_loss"],
            w1_spectral["total_w1"], w1_spectral["low_band_w1"], w1_spectral["high_band_w1"],
            time_spectral,
        )
        logger.info(
            "    coherence: ref_neighbor=%.3f gen_neighbor=%.3f ref_mode=%.3f gen_mode=%.3f",
            coherence["ref_neighbor_corr"], coherence["gen_neighbor_corr"],
            coherence["ref_mode_energy_corr"], coherence["gen_mode_energy_corr"],
        )

        per_seed.append(seed_entry)

    # Statistical analysis: paired t-tests with Bonferroni correction
    u = np.array(uniform_w1s)
    b = np.array(band_w1s)
    s = np.array(spectral_w1s)

    # Bonferroni alpha = 0.05 / 4 families = 0.0125
    bonferroni_alpha = 0.0125
    comparisons: dict[str, dict] = {}

    for name, pair in [
        ("spectral_vs_uniform", (u, s)),
        ("spectral_vs_band", (b, s)),
        ("band_vs_uniform", (u, b)),
    ]:
        a_arr, b_arr = pair
        if len(a_arr) >= 2:
            t_stat, p_val = ttest_rel(a_arr, b_arr)
        else:
            t_stat, p_val = 0.0, 1.0
        diff = a_arr - b_arr
        improvement_pct = float(diff.mean() / a_arr.mean() * 100) if a_arr.mean() > 0 else 0.0
        comparisons[name] = {
            "t_stat": float(t_stat),
            "p_val": float(p_val),
            "improvement_pct": improvement_pct,
            "significant": bool(p_val < bonferroni_alpha),
        }

    logger.info("\n  %s SUMMARY:", family)
    logger.info("    Uniform W1:  %.1f +/- %.1f", u.mean(), u.std())
    logger.info("    Band W1:     %.1f +/- %.1f", b.mean(), b.std())
    logger.info("    Spectral W1: %.1f +/- %.1f", s.mean(), s.std())
    for cname, cdata in comparisons.items():
        sig = "SIGNIFICANT" if cdata["significant"] else "not significant"
        logger.info(
            "    %s: %.1f%% improvement, t=%.3f, p=%.4f (%s)",
            cname, cdata["improvement_pct"], cdata["t_stat"], cdata["p_val"], sig,
        )

    return {
        "family": family,
        "n_nodes": n_nodes,
        "uniform_w1s": uniform_w1s,
        "band_w1s": band_w1s,
        "spectral_w1s": spectral_w1s,
        "uniform_w1_mean": float(u.mean()),
        "uniform_w1_std": float(u.std()),
        "band_w1_mean": float(b.mean()),
        "band_w1_std": float(b.std()),
        "spectral_w1_mean": float(s.mean()),
        "spectral_w1_std": float(s.std()),
        "comparisons": comparisons,
        "per_seed": per_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 5a: Independent per-mode spectral diffusion vs spatial baselines"
    )
    parser.add_argument(
        "--families", type=str, default=DEFAULT_FAMILIES,
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
        "--n-epochs", type=int, default=500,
        help="Training epochs per run (default: 500)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device: 'cuda' or 'cpu' (default: cuda)",
    )
    parser.add_argument(
        "--output", type=str, default="results/phase5a/phase5a_results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--dataset-dir", type=str, default="results/phase5a/datasets",
        help="Dataset cache directory (separate per n_nodes to prevent collisions)",
    )
    parser.add_argument(
        "--energy-exponent", type=float, default=0.5,
        help="Energy exponent for per-mode t_max schedule (default: 0.5)",
    )
    parser.add_argument(
        "--batch-timesteps", type=int, default=32,
        help="Batch timesteps per epoch for spatial methods (default: 32)",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=128,
        help="Hidden dimension for all networks (default: 128)",
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
            energy_exponent=args.energy_exponent,
        )
        all_results.append(result)

        # Crash-safe: write after each family
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("  Saved intermediate results to %s", output_path)

    # Summary table
    print("\n" + "=" * 100)
    print(f"{'Family':<20} {'Unif W1':>10} {'Band W1':>10} {'Spec W1':>10} "
          f"{'Spec vs Unif':>14} {'Spec vs Band':>14} {'p(vs Band)':>12}")
    print("-" * 100)
    for r in all_results:
        c_su = r["comparisons"]["spectral_vs_uniform"]
        c_sb = r["comparisons"]["spectral_vs_band"]
        sig_sb = "*" if c_sb["significant"] else " "
        print(
            f"{r['family']:<20} "
            f"{r['uniform_w1_mean']:>10.1f} "
            f"{r['band_w1_mean']:>10.1f} "
            f"{r['spectral_w1_mean']:>10.1f} "
            f"{c_su['improvement_pct']:>12.1f}% "
            f"{c_sb['improvement_pct']:>12.1f}% "
            f"{c_sb['p_val']:>10.4f}{sig_sb:>2}"
        )
    print("=" * 100)
    print(f"\nBonferroni alpha = 0.0125 (4 families)")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
