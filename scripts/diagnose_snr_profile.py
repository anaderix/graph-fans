#!/usr/bin/env python
"""NS-D: Measure per-SNR-bin loss to confirm gradient misallocation.

Trains a model with the current default settings (ε-prediction, uniform
t-sampling, cosine schedule) and then evaluates MSE loss across timesteps
binned by log10(SNR).  The gradient_share_uniform_t column shows how much
training gradient each SNR bin receives under uniform t-sampling, making
misallocation explicit.

Usage:
    uv run python scripts/diagnose_snr_profile.py \\
        --dataset-dir results/phase2f/datasets \\
        --family "SBM(q=0.05)" --seed 0 \\
        --n-nodes 200 --epochs 500 --device cpu \\
        --n-snr-bins 10

Output:
    results/diagnostics/snr_profile_<family>_seed<seed>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `graph_fans` is importable when
# this script is run directly (e.g. `uv run python scripts/diagnose_snr_profile.py`).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from graph_fans.phase2.dataset import load_dataset
from graph_fans.phase2.evaluate import _get_graph
from graph_fans.phase2.sde import CosineScheduleSDE
from graph_fans.phase2.trainer import Trainer, TrainConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Number of timestep points for the evaluation sweep
N_EVAL_T_POINTS = 1000

# Number of (x_0, noise) pairs to average per timestep during evaluation
N_EVAL_PAIRS_PER_T = 10


def _safe_log10_snr(alpha_bar: float) -> float:
    """log10(SNR) = log10(alpha_bar / (1 - alpha_bar)), clamped for stability."""
    ab = float(np.clip(alpha_bar, 1e-6, 1.0 - 1e-6))
    snr = ab / (1.0 - ab)
    return float(np.log10(max(snr, 1e-10)))


def compute_snr_profile(
    trainer: Trainer,
    n_t_points: int,
    n_pairs_per_t: int,
    n_snr_bins: int,
    seed: int,
) -> dict:
    """Evaluate per-timestep MSE loss and bin by log10(SNR).

    Uses training data only — we want to know whether the model learned
    the training distribution, not whether it generalises.  EMA weights
    are used (same as generation).

    Args:
        trainer: Trained Trainer instance.  EMA weights are applied
            temporarily during evaluation.
        n_t_points: Number of evenly-spaced timesteps to sweep.
        n_pairs_per_t: Number of (x_0, noise) pairs to average per t.
        n_snr_bins: Number of log-SNR bins.
        seed: RNG seed for reproducibility.

    Returns:
        Dict with per-timestep records and per-bin summary.
    """
    rng = np.random.RandomState(seed + 9999)   # different from training seed

    sde: CosineScheduleSDE = trainer.sde  # type: ignore[assignment]
    model = trainer.model
    device = trainer.device
    edge_index = trainer.edge_index
    dataset = trainer.dataset          # [N, n_nodes, n_features]
    n_samples = trainer.n_samples

    # Apply EMA weights for evaluation
    original_weights = None
    if trainer.ema is not None:
        original_weights = trainer.ema.apply(model)
    model.eval()

    t_min = 1e-5
    t_max = sde.T
    ts = np.linspace(t_min, t_max, n_t_points)

    per_t_records: list[dict] = []

    with torch.no_grad():
        for t in ts:
            ab = sde.alpha_bar(t)
            log_snr = _safe_log10_snr(ab)

            mse_values = []
            for _ in range(n_pairs_per_t):
                idx = int(rng.randint(0, n_samples))
                x_0 = dataset[idx]                                  # [n_nodes, n_features]
                noise = torch.randn_like(x_0)
                x_t, _ = sde.perturb(x_0, float(t), noise=noise)

                t_tensor = torch.tensor([float(t)], dtype=torch.float32, device=device)
                eps_pred = model(x_t, t_tensor, edge_index)

                mse = float(torch.nn.functional.mse_loss(eps_pred, noise).item())
                mse_values.append(mse)

            per_t_records.append({
                "t": float(t),
                "alpha_bar": float(ab),
                "log10_snr": log_snr,
                "mean_mse": float(np.mean(mse_values)),
            })

    # Restore original (non-EMA) weights
    if trainer.ema is not None and original_weights is not None:
        trainer.ema.restore(model, original_weights)
    model.train()

    # ---- Bin by log10(SNR) -----------------------------------------------
    log_snr_values = np.array([r["log10_snr"] for r in per_t_records])
    mse_values_all = np.array([r["mean_mse"] for r in per_t_records])

    snr_min = log_snr_values.min()
    snr_max = log_snr_values.max()
    bin_edges = np.linspace(snr_min, snr_max, n_snr_bins + 1)

    bins: list[dict] = []
    for i in range(n_snr_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]

        # Include right edge only for the last bin
        if i < n_snr_bins - 1:
            mask = (log_snr_values >= lo) & (log_snr_values < hi)
        else:
            mask = (log_snr_values >= lo) & (log_snr_values <= hi)

        n_in_bin = int(mask.sum())
        mean_log_snr = float(log_snr_values[mask].mean()) if n_in_bin > 0 else float("nan")
        mean_loss = float(mse_values_all[mask].mean()) if n_in_bin > 0 else float("nan")

        # Gradient share under uniform t: fraction of [t_min, T] interval
        # that maps to this SNR bin.  We approximate as (# timesteps in bin
        # / total timesteps) since t is sampled uniformly in [t_min, T].
        gradient_share = float(n_in_bin) / float(n_t_points)

        bins.append({
            "bin_index": i,
            "log10_snr_lo": float(lo),
            "log10_snr_hi": float(hi),
            "mean_log10_snr": mean_log_snr,
            "mean_loss": mean_loss,
            "n_t_points": n_in_bin,
            "gradient_share_uniform_t": gradient_share,
        })

    return {
        "per_t_records": per_t_records,
        "bins": bins,
        "n_t_points": n_t_points,
        "n_pairs_per_t": n_pairs_per_t,
        "n_snr_bins": n_snr_bins,
    }


def print_table(bins: list[dict], family: str, seed: int, epochs: int) -> None:
    """Print a human-readable table of SNR-bin diagnostics."""
    print()
    print("=" * 90)
    print(f"SNR-PROFILE DIAGNOSTIC: {family}, seed={seed}, {epochs} epochs")
    print("=" * 90)
    header = (
        f"{'bin':>4}  {'log10(SNR) range':>22}  "
        f"{'mean_log10(SNR)':>15}  {'mean_loss':>10}  "
        f"{'n_t_pts':>7}  {'grad_share':>10}"
    )
    print(header)
    print("-" * 90)
    for b in bins:
        lo = b["log10_snr_lo"]
        hi = b["log10_snr_hi"]
        rng_str = f"[{lo:+.2f}, {hi:+.2f})"
        mean_snr = b["mean_log10_snr"]
        loss = b["mean_loss"]
        n_pts = b["n_t_points"]
        gshare = b["gradient_share_uniform_t"]

        if np.isnan(loss):
            loss_str = "      nan"
        else:
            loss_str = f"{loss:10.5f}"

        if np.isnan(mean_snr):
            snr_str = "             nan"
        else:
            snr_str = f"{mean_snr:+15.3f}"

        print(
            f"{b['bin_index']:>4}  {rng_str:>22}  "
            f"{snr_str}  {loss_str}  "
            f"{n_pts:>7}  {gshare:>10.4f}"
        )
    print("-" * 90)
    print(
        "grad_share: fraction of uniform-t gradient that falls in this SNR bin.\n"
        "High grad_share + low loss → wasted capacity.  Low grad_share + high loss → underfit."
    )
    print()


def run_diagnostic(
    dataset_dir: str,
    family: str,
    seed: int,
    n_nodes: int,
    epochs: int,
    device: str,
    n_snr_bins: int,
    output_dir: str,
    t_sampling: str = "uniform",
    min_snr_gamma: float | None = None,
    hidden_dim: int = 128,
    n_layers: int = 3,
    conv_type: str = "gcn",
) -> dict:
    """Full pipeline: load data, train, evaluate, save.

    Args:
        dataset_dir: Path to directory containing .npz dataset files.
        family: Graph family string (e.g. "SBM(q=0.05)").
        seed: Experiment seed for dataset loading and training.
        n_nodes: Number of nodes (used to build graph topology).
        epochs: Number of training epochs.
        device: PyTorch device string.
        n_snr_bins: Number of log-SNR bins.
        output_dir: Where to write the output JSON.

    Returns:
        Full results dict that is also serialised to JSON.
    """
    # ---- Load dataset --------------------------------------------------------
    safe_family = family.replace("(", "_").replace(")", "").replace("=", "")
    ds_path = Path(dataset_dir) / f"{safe_family}_seed{seed}.npz"

    if not ds_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {ds_path}\n"
            "Re-run with a valid --dataset-dir or generate the dataset first."
        )

    ds = load_dataset(ds_path)
    train_data = ds["train"]        # [N, n_nodes, n_features]
    logger.info(f"Loaded dataset: train={train_data.shape}, family={family}, seed={seed}")

    # ---- Build graph topology ------------------------------------------------
    graph = _get_graph(family, n_nodes, seed=0)   # topology seed is always 0

    # ---- Train ---------------------------------------------------------------
    n_train_samples = train_data.shape[0]
    cfg = TrainConfig(
        n_epochs=epochs,
        batch_timesteps=32,
        seed=seed,
        device=device,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        conv_type=conv_type,
        n_gen_steps=200,
        sde_type="cosine",
        use_ema=True,
        ema_decay=0.999,
        use_lr_scheduler=True,
        use_spectral_noise=False,
        use_spectral_loss=False,
        n_train_samples=n_train_samples,
        t_sampling=t_sampling,
        min_snr_gamma=min_snr_gamma,
    )

    logger.info(f"Training for {epochs} epochs on {device} ...")
    trainer = Trainer(cfg, graph, train_data, importance_weights=None)
    history = trainer.train()

    final_loss = history["loss"][-1]
    logger.info(f"Training complete. Final loss = {final_loss:.6f}")

    # ---- Evaluate per-SNR-bin loss -------------------------------------------
    logger.info(
        f"Evaluating loss at {N_EVAL_T_POINTS} timesteps "
        f"x {N_EVAL_PAIRS_PER_T} pairs per t ..."
    )
    # Use a different seed for evaluation to avoid overlap with training RNG state
    profile = compute_snr_profile(
        trainer=trainer,
        n_t_points=N_EVAL_T_POINTS,
        n_pairs_per_t=N_EVAL_PAIRS_PER_T,
        n_snr_bins=n_snr_bins,
        seed=seed,
    )

    # ---- Print table ---------------------------------------------------------
    print_table(profile["bins"], family, seed, epochs)

    # ---- Save JSON -----------------------------------------------------------
    results = {
        "family": family,
        "seed": seed,
        "n_nodes": n_nodes,
        "epochs": epochs,
        "device": device,
        "n_snr_bins": n_snr_bins,
        "final_training_loss": final_loss,
        "training_loss_history": history["loss"],
        "sde_type": "cosine",
        "t_sampling": t_sampling,
        "min_snr_gamma": min_snr_gamma,
        "prediction_target": "epsilon",
        "n_eval_t_points": N_EVAL_T_POINTS,
        "n_eval_pairs_per_t": N_EVAL_PAIRS_PER_T,
        "bins": profile["bins"],
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    suffix = f"_{conv_type}_{t_sampling}_L{n_layers}"
    if min_snr_gamma is not None:
        suffix += f"_gamma{min_snr_gamma}"
    json_path = out_path / f"snr_profile_{safe_family}_seed{seed}{suffix}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {json_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NS-D: per-SNR-bin loss diagnostic for gradient misallocation"
    )
    parser.add_argument(
        "--dataset-dir",
        default="results/phase2f/datasets",
        help="Directory containing .npz dataset files (default: results/phase2f/datasets)",
    )
    parser.add_argument(
        "--family",
        default="SBM(q=0.05)",
        help='Graph family string, e.g. "SBM(q=0.05)" or "BA(m=5)"',
    )
    parser.add_argument("--seed", type=int, default=0, help="Dataset and training seed")
    parser.add_argument(
        "--n-nodes",
        type=int,
        default=200,
        help="Number of nodes for graph topology generation (default: 200)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of training epochs (default: 500)",
    )
    parser.add_argument("--device", default="cpu", help="PyTorch device (default: cpu)")
    parser.add_argument(
        "--n-snr-bins",
        type=int,
        default=10,
        help="Number of log-SNR bins (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/diagnostics",
        help="Output directory for JSON results (default: results/diagnostics)",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=128,
        help="Score network hidden dimension (default: 128)",
    )
    parser.add_argument(
        "--n-layers", type=int, default=3,
        help="Score network GCN layers (default: 3)",
    )
    parser.add_argument(
        "--conv-type", choices=["gcn", "transformer"], default="gcn",
        help="Graph convolution type (default: gcn)",
    )
    parser.add_argument(
        "--t-sampling",
        choices=["uniform", "log_snr"],
        default="uniform",
        help="Timestep sampling strategy (default: uniform)",
    )
    parser.add_argument(
        "--min-snr-gamma",
        type=float,
        default=None,
        help="min-SNR-gamma loss weighting (default: None=disabled, 5.0=standard)",
    )
    args = parser.parse_args()

    run_diagnostic(
        dataset_dir=args.dataset_dir,
        family=args.family,
        seed=args.seed,
        n_nodes=args.n_nodes,
        epochs=args.epochs,
        device=args.device,
        n_snr_bins=args.n_snr_bins,
        output_dir=args.output_dir,
        t_sampling=args.t_sampling,
        min_snr_gamma=args.min_snr_gamma,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        conv_type=args.conv_type,
    )


if __name__ == "__main__":
    main()
