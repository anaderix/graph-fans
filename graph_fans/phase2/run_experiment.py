"""Main script for Phase 2: Regime A Core (H1-A).

Usage:
    uv run python -m graph_fans.phase2 [options]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")

from .evaluate import run_h1a_experiment, compute_h1a_decision
from .trainer import TrainConfig
from .visualize import (
    plot_per_band_comparison,
    plot_h1a_summary,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_phase2(
    output_dir: str = "results/phase2",
    n_nodes: int = 200,
    n_features: int = 16,
    B: int = 8,
    n_seeds: int = 5,
    n_epochs: int = 2000,
    batch_timesteps: int = 32,
    h1a_families: list[str] | None = None,
    device: str = "cpu",
    sde_type: str = "cosine",
    use_ema: bool = True,
    use_lr_scheduler: bool = True,
    use_spectral_loss: bool = False,
    spectral_loss_weight: float = 0.1,
    feature_mode: str = "community",
    n_train_samples: int = 500,
    dataset_dir: str = "results/phase2/datasets",
    pre_generate_only: bool = False,
    t_sampling: str = "uniform",
    min_snr_gamma: float | None = None,
    hidden_dim: int = 128,
    n_layers: int = 3,
    conv_type: str = "gcn",
) -> dict:
    """Run Phase 2 experiment pipeline (H1-A only)."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    config = TrainConfig(
        n_epochs=n_epochs,
        batch_timesteps=batch_timesteps,
        device=device,
        B=B,
        sde_type=sde_type,
        use_ema=use_ema,
        use_lr_scheduler=use_lr_scheduler,
        use_spectral_loss=use_spectral_loss,
        spectral_loss_weight=spectral_loss_weight,
        n_train_samples=n_train_samples,
        t_sampling=t_sampling,
        min_snr_gamma=min_snr_gamma,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        conv_type=conv_type,
    )

    logger.info(f"Config: sde={sde_type}, ema={use_ema}, lr_scheduler={use_lr_scheduler}, "
                f"spectral_loss={use_spectral_loss}, features={feature_mode}, "
                f"n_train_samples={n_train_samples}, epochs={n_epochs}, batch_timesteps={batch_timesteps}")

    # Pre-generate datasets if requested
    if pre_generate_only:
        from .dataset import get_or_generate_dataset, validate_dataset
        from .evaluate import _get_graph
        all_families = set(h1a_families or ["SBM(q=0.05)", "BA(m=5)"])
        logger.info(f"Pre-generating datasets for {len(all_families)} families × {n_seeds} seeds...")
        for family in sorted(all_families):
            graph = _get_graph(family, n_nodes, seed=0)
            for seed in range(n_seeds):
                ds = get_or_generate_dataset(
                    graph, family, seed,
                    n_train=n_train_samples, n_ref=50,
                    n_features=n_features, feature_mode=feature_mode,
                    cache_dir=dataset_dir,
                )
                val = validate_dataset(ds, graph, B)
                logger.info(f"  {family} seed={seed}: train_std={val['train_std']:.3f}, "
                            f"profile={np.array2string(val['spectral_profile'], precision=3)}")
        logger.info(f"Datasets saved to {dataset_dir}/. Inspect and re-run without --pre-generate-only.")
        return {}

    # --- H1-A Experiment ---
    logger.info("\n=== H1-A: Uniform vs Spectral Noise ===")
    h1a_df = run_h1a_experiment(
        families=h1a_families,
        n_seeds=n_seeds,
        n_nodes=n_nodes,
        n_features=n_features,
        config=config,
        B=B,
        output_dir=output_dir,
        feature_mode=feature_mode,
        dataset_dir=dataset_dir,
    )

    logger.info("\nH1-A Results Summary:")
    summary = h1a_df.groupby(["family", "method"])[["qbe_high_bands", "qbe_total"]].agg(["mean", "std"])
    logger.info(f"\n{summary}")

    # --- H1-A Decision ---
    logger.info("\n=== H1-A Decision ===")
    g2 = compute_h1a_decision(h1a_df, B, output_dir)
    logger.info(f"Decision: {g2['decision']}")
    logger.info(
        f"H1-A: {g2['h1a']['families_with_improvement']}/{g2['h1a']['total_families']} families improved"
    )

    # --- Plots ---
    logger.info("\nGenerating plots...")
    plot_per_band_comparison(h1a_df, B, save_path=out_path / "per_band_comparison")
    plot_h1a_summary(h1a_df, save_path=out_path / "h1a_summary")

    logger.info(f"\nResults saved to {out_path}/")
    return g2


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Regime A Core (H1-A)")
    parser.add_argument("--output-dir", default="results/phase2")
    parser.add_argument("--n-nodes", type=int, default=200)
    parser.add_argument("--n-features", type=int, default=16)
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-timesteps", type=int, default=32,
                        help="Number of timesteps per epoch (default: 32)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sde", choices=["vpsde", "cosine"], default="vpsde",
                        help="SDE type: vpsde (linear beta) or cosine (Nichol & Dhariwal)")
    parser.add_argument("--ema", action="store_true", help="Use EMA of model weights")
    parser.add_argument("--lr-scheduler", action="store_true", help="Use cosine LR annealing")
    parser.add_argument("--spectral-loss", action="store_true",
                        help="Alt-4: add spectral fidelity loss term")
    parser.add_argument("--spectral-loss-weight", type=float, default=0.1,
                        help="Weight for spectral loss term (default: 0.1)")
    parser.add_argument("--feature-mode", choices=["smooth", "multiscale", "community"],
                        default="community", help="Feature generation mode")
    parser.add_argument("--n-train-samples", type=int, default=500,
                        help="Number of feature realizations per graph for training")
    parser.add_argument("--dataset-dir", default="results/phase2/datasets",
                        help="Directory for cached feature datasets")
    parser.add_argument("--pre-generate-only", action="store_true",
                        help="Generate datasets and exit (for inspection before training)")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Score network hidden dimension (default: 128)")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="Score network GCN layers (default: 3)")
    parser.add_argument("--conv-type", choices=["gcn", "transformer"], default="gcn",
                        help="Graph convolution type (default: gcn)")
    parser.add_argument("--t-sampling", choices=["uniform", "log_snr"], default="uniform",
                        help="Timestep sampling strategy: uniform (default) or log_snr (NS-A)")
    parser.add_argument("--min-snr-gamma", type=float, default=None,
                        help="min-SNR-γ loss weighting (NS-C). None=disabled, 5.0=standard")
    args = parser.parse_args()

    run_phase2(
        output_dir=args.output_dir,
        n_nodes=args.n_nodes,
        n_features=args.n_features,
        B=args.bands,
        n_seeds=args.seeds,
        n_epochs=args.epochs,
        batch_timesteps=args.batch_timesteps,
        device=args.device,
        sde_type=args.sde,
        use_ema=args.ema,
        use_lr_scheduler=args.lr_scheduler,
        use_spectral_loss=getattr(args, "spectral_loss", False),
        spectral_loss_weight=getattr(args, "spectral_loss_weight", 0.1),
        feature_mode=args.feature_mode,
        n_train_samples=args.n_train_samples,
        dataset_dir=args.dataset_dir,
        pre_generate_only=args.pre_generate_only,
        t_sampling=args.t_sampling,
        min_snr_gamma=args.min_snr_gamma,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        conv_type=args.conv_type,
    )


if __name__ == "__main__":
    main()
