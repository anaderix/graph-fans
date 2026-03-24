"""Evaluation pipeline for Phase 2: H1-A and H2 experiments.

Fixed: trains on a DATASET of feature realizations (not a single sample),
evaluates generated distribution against held-out distribution.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from graph_fans.phase0.spectral_profiler import (
    compute_laplacian_spectrum,
    partition_into_bands,
    compute_band_energy,
)
from graph_fans.utils.graph_generators import generate_sbm, generate_ba, load_citation_network
from .dataset import get_or_generate_dataset, validate_dataset
from .noise_shaper import compute_importance_weights, ImportanceWeights
from .trainer import Trainer, TrainConfig

logger = logging.getLogger(__name__)

N_GEN_SAMPLES = 50  # number of samples to generate for evaluation


@dataclass
class Phase2Results:
    """Results for a single experiment run."""

    graph_family: str
    method: str
    seed: int
    per_band_qbe_diff: np.ndarray  # [B] per-band mean energy diff (gen vs ref distribution)
    qbe_high_bands: float
    qbe_total: float
    final_loss: float
    train_time_seconds: float


def evaluate_distribution(
    graph: nx.Graph,
    features_ref: np.ndarray,       # [N_ref, n_nodes, n_features]
    features_gen: np.ndarray,       # [N_gen, n_nodes, n_features]
    graph_family: str,
    method: str,
    seed: int,
    B: int = 8,
    final_loss: float = 0.0,
    train_time: float = 0.0,
) -> Phase2Results:
    """Evaluate generated feature DISTRIBUTION against reference distribution.

    Computes mean spectral energy profile across samples, then measures
    per-band distance between the two distributions' profiles.
    """
    eigenvalues, eigenvectors = compute_laplacian_spectrum(graph)
    _, band_indices = partition_into_bands(eigenvalues, B)

    def mean_band_profile(dataset: np.ndarray) -> np.ndarray:
        """Compute mean normalized band energy across samples."""
        profiles = []
        for features in dataset:
            energy = compute_band_energy(features, eigenvectors, band_indices)
            total = energy.sum()
            profiles.append(energy / total if total > 0 else energy)
        return np.stack(profiles).mean(axis=0)

    profile_ref = mean_band_profile(features_ref)
    profile_gen = mean_band_profile(features_gen)

    per_band_diff = np.abs(profile_ref - profile_gen)
    half = B // 2
    qbe_high = float(per_band_diff[half:].mean())
    qbe_total = float(np.linalg.norm(profile_ref - profile_gen))

    return Phase2Results(
        graph_family=graph_family,
        method=method,
        seed=seed,
        per_band_qbe_diff=per_band_diff,
        qbe_high_bands=qbe_high,
        qbe_total=qbe_total,
        final_loss=final_loss,
        train_time_seconds=train_time,
    )


def _load_phase0_energies(phase0_dir: str = "results/phase0") -> dict[str, np.ndarray]:
    """Load band energies from Phase 0 results."""
    path = Path(phase0_dir) / "band_energies.json"
    if not path.exists():
        logger.warning(f"Phase 0 energies not found at {path}, using uniform weights")
        return {}
    with open(path) as f:
        data = json.load(f)
    return {name: np.array(v["band_energies"]) for name, v in data.items()}


def _get_graph(family: str, n_nodes: int, seed: int = 0) -> nx.Graph:
    """Get graph topology for a family (features generated separately as dataset)."""
    if family.startswith("SBM"):
        q = float(family.split("=")[1].rstrip(")"))
        gd = generate_sbm(n_nodes=n_nodes, p_inter=q, seed=seed, feature_mode="smooth")
    elif family.startswith("BA"):
        m = int(family.split("=")[1].rstrip(")"))
        gd = generate_ba(n_nodes=n_nodes, m=m, seed=seed, feature_mode="smooth")
    elif family == "Cora":
        gd = load_citation_network("Cora")
    else:
        raise ValueError(f"Unknown family: {family}")
    return gd.graph


def _get_importance_weights(
    phase0_energies: dict[str, np.ndarray],
    family: str,
    alpha: float = 1.0,
    epsilon: float = 1e-3,
) -> ImportanceWeights | None:
    """Get importance weights for a family from Phase 0 data."""
    if family in phase0_energies:
        return compute_importance_weights(phase0_energies[family], alpha, epsilon)
    for key in phase0_energies:
        if family.lower() in key.lower() or key.lower() in family.lower():
            return compute_importance_weights(phase0_energies[key], alpha, epsilon)
    logger.warning(f"No Phase 0 energies for {family}, using uniform weights")
    return None


def run_single_experiment(
    family: str,
    method: str,
    seed: int,
    n_nodes: int = 200,
    n_features: int = 16,
    config: TrainConfig | None = None,
    importance_weights: ImportanceWeights | None = None,
    B: int = 8,
    feature_mode: str = "community",
    dataset_dir: str | Path = "results/phase2/datasets",
) -> Phase2Results:
    """Run a single training + generation + evaluation."""
    if config is None:
        config = TrainConfig()

    # Fixed graph topology
    graph = _get_graph(family, n_nodes, seed=0)

    # Load or generate cached dataset
    dataset = get_or_generate_dataset(
        graph, family, seed,
        n_train=config.n_train_samples,
        n_ref=N_GEN_SAMPLES,
        n_features=n_features,
        feature_mode=feature_mode,
        cache_dir=dataset_dir,
    )
    train_dataset = dataset["train"]
    ref_dataset = dataset["ref"]

    # Configure
    cfg = TrainConfig(
        n_epochs=config.n_epochs,
        lr=config.lr,
        batch_timesteps=config.batch_timesteps,
        seed=seed,
        device=config.device,
        use_spectral_noise=(method != "uniform"),
        t_knee=None,
        alpha=config.alpha,
        epsilon=config.epsilon,
        B=B,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        n_gen_steps=config.n_gen_steps,
        sde_type=config.sde_type,
        use_ema=config.use_ema,
        ema_decay=config.ema_decay,
        use_lr_scheduler=config.use_lr_scheduler,
        use_spectral_loss=config.use_spectral_loss,
        spectral_loss_weight=config.spectral_loss_weight,
        n_train_samples=config.n_train_samples,
    )

    if "tknee=" in method:
        cfg.t_knee = float(method.split("tknee=")[1])
        cfg.use_spectral_noise = True

    logger.info(f"  Training {family} / {method} / seed={seed} ({config.n_train_samples} samples)")
    start = time.time()

    trainer = Trainer(cfg, graph, train_dataset, importance_weights)
    history = trainer.train()
    train_time = time.time() - start

    # Sanity check
    sanity = trainer.sanity_check(train_dataset)
    if sanity["warnings"]:
        logger.warning(f"  Sanity check warnings for {family}/{method}/seed={seed}: {sanity['warnings']}")

    # Generate samples
    logger.info(f"  Generating {N_GEN_SAMPLES} samples...")
    gen_dataset = trainer.generate(n_samples=N_GEN_SAMPLES)

    final_loss = history["loss"][-1]

    return evaluate_distribution(
        graph, ref_dataset, gen_dataset,
        family, method, seed, B,
        final_loss, train_time,
    )


def run_h1a_experiment(
    families: list[str] | None = None,
    n_seeds: int = 5,
    n_nodes: int = 200,
    n_features: int = 16,
    config: TrainConfig | None = None,
    B: int = 8,
    output_dir: str = "results/phase2",
    feature_mode: str = "community",
    dataset_dir: str | Path = "results/phase2/datasets",
) -> pd.DataFrame:
    """Run H1-A: uniform vs spectral noise on primary families."""
    if families is None:
        families = ["SBM(q=0.05)", "BA(m=5)"]
    if config is None:
        config = TrainConfig()

    # Pre-generate all datasets
    logger.info("Pre-generating all H1-A datasets...")
    for family in families:
        graph = _get_graph(family, n_nodes, seed=0)
        for seed in range(n_seeds):
            ds = get_or_generate_dataset(
                graph, family, seed,
                n_train=config.n_train_samples, n_ref=N_GEN_SAMPLES,
                n_features=n_features, feature_mode=feature_mode,
                cache_dir=dataset_dir,
            )
            validation = validate_dataset(ds, graph, B)
            if validation["warnings"]:
                logger.warning(f"  {family} seed={seed}: {validation['warnings']}")

    phase0_energies = _load_phase0_energies()
    results = []

    for family in families:
        weights = _get_importance_weights(phase0_energies, family, config.alpha, config.epsilon)

        for seed in range(n_seeds):
            for method in ["uniform", "spectral"]:
                r = run_single_experiment(
                    family, method, seed, n_nodes, n_features, config,
                    weights, B, feature_mode, dataset_dir,
                )
                results.append(r)

    rows = []
    for r in results:
        row = {
            "family": r.graph_family,
            "method": r.method,
            "seed": r.seed,
            "qbe_high_bands": r.qbe_high_bands,
            "qbe_total": r.qbe_total,
            "final_loss": r.final_loss,
            "train_time_s": r.train_time_seconds,
        }
        for b in range(len(r.per_band_qbe_diff)):
            row[f"qbe_band_{b}"] = r.per_band_qbe_diff[b]
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path / "h1a_results.csv", index=False)
    logger.info(f"H1-A results saved to {out_path / 'h1a_results.csv'}")
    return df


def run_h2_experiment(
    families: list[str] | None = None,
    t_knee_values: list[float] | None = None,
    n_seeds: int = 5,
    n_nodes: int = 200,
    n_features: int = 16,
    config: TrainConfig | None = None,
    B: int = 8,
    output_dir: str = "results/phase2",
    feature_mode: str = "community",
    dataset_dir: str | Path = "results/phase2/datasets",
) -> pd.DataFrame:
    """Run H2: grid search over t_knee."""
    if families is None:
        families = ["SBM(q=0.01)", "SBM(q=0.05)", "SBM(q=0.1)", "BA(m=2)", "BA(m=5)"]
    if t_knee_values is None:
        t_knee_values = [0.05, 0.10, 0.15, 0.20, 0.30]
    if config is None:
        config = TrainConfig()

    # Pre-generate all datasets
    logger.info("Pre-generating all H2 datasets...")
    for family in families:
        graph = _get_graph(family, n_nodes, seed=0)
        for seed in range(n_seeds):
            get_or_generate_dataset(
                graph, family, seed,
                n_train=config.n_train_samples, n_ref=N_GEN_SAMPLES,
                n_features=n_features, feature_mode=feature_mode,
                cache_dir=dataset_dir,
            )

    phase0_energies = _load_phase0_energies()
    results = []

    for family in families:
        weights = _get_importance_weights(phase0_energies, family, config.alpha, config.epsilon)

        graph = _get_graph(family, n_nodes, seed=0)
        eigenvalues, _ = compute_laplacian_spectrum(graph)
        lambda_2 = eigenvalues[1] if len(eigenvalues) > 1 else 0
        lambda_max = eigenvalues[-1] if eigenvalues[-1] > 0 else 1
        spectral_gap_ratio = lambda_2 / lambda_max

        for t_knee in t_knee_values:
            for seed in range(n_seeds):
                method = f"spectral_ramp_tknee={t_knee}"
                r = run_single_experiment(
                    family, method, seed, n_nodes, n_features, config,
                    weights, B, feature_mode, dataset_dir,
                )
                results.append((r, spectral_gap_ratio, t_knee))

    rows = []
    for r, sgr, tk in results:
        rows.append({
            "family": r.graph_family,
            "t_knee": tk,
            "seed": r.seed,
            "spectral_gap_ratio": sgr,
            "qbe_total": r.qbe_total,
            "qbe_high_bands": r.qbe_high_bands,
            "final_loss": r.final_loss,
            "train_time_s": r.train_time_seconds,
        })

    df = pd.DataFrame(rows)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path / "h2_results.csv", index=False)
    logger.info(f"H2 results saved to {out_path / 'h2_results.csv'}")
    return df


def compute_g2_decision(
    h1a_df: pd.DataFrame,
    h2_df: pd.DataFrame,
    B: int = 8,
    output_dir: str = "results/phase2",
) -> dict:
    """Compute G2 go/no-go decision."""
    from scipy.stats import ttest_rel

    families = h1a_df["family"].unique()
    h1a_pass_count = 0
    h1a_details = {}

    for family in families:
        uniform = h1a_df[(h1a_df["family"] == family) & (h1a_df["method"] == "uniform")]
        spectral = h1a_df[(h1a_df["family"] == family) & (h1a_df["method"] == "spectral")]

        if len(uniform) == 0 or len(spectral) == 0:
            continue

        u_vals = uniform["qbe_high_bands"].values
        s_vals = spectral["qbe_high_bands"].values
        n = min(len(u_vals), len(s_vals))
        u_vals, s_vals = u_vals[:n], s_vals[:n]

        improvement = u_vals.mean() - s_vals.mean()
        if n >= 2:
            t_stat, p_val = ttest_rel(u_vals, s_vals)
        else:
            t_stat, p_val = 0.0, 1.0

        adjusted_alpha = 0.05 / len(families)
        passes = improvement > 0 and p_val < adjusted_alpha

        if passes:
            h1a_pass_count += 1

        h1a_details[family] = {
            "improvement": float(improvement),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "adjusted_alpha": adjusted_alpha,
            "passes_significance": bool(passes),
            "uniform_mean": float(u_vals.mean()),
            "spectral_mean": float(s_vals.mean()),
        }

    h1a_pass = h1a_pass_count >= 2

    # H2
    optimal_tknee = {}
    spectral_gaps = {}

    for family in h2_df["family"].unique():
        fam_df = h2_df[h2_df["family"] == family]
        mean_by_tknee = fam_df.groupby("t_knee")["qbe_total"].mean()
        best_tknee = mean_by_tknee.idxmin()
        optimal_tknee[family] = best_tknee
        spectral_gaps[family] = fam_df["spectral_gap_ratio"].iloc[0]

    if len(optimal_tknee) >= 3:
        families_ordered = sorted(optimal_tknee.keys())
        tknee_vals = [optimal_tknee[f] for f in families_ordered]
        sgr_vals = [spectral_gaps[f] for f in families_ordered]

        rho, p_val_h2 = spearmanr(sgr_vals, tknee_vals)

        rng = np.random.RandomState(42)
        rhos_boot = []
        for _ in range(10000):
            idx = rng.choice(len(tknee_vals), size=len(tknee_vals), replace=True)
            r, _ = spearmanr([sgr_vals[i] for i in idx], [tknee_vals[i] for i in idx])
            if not np.isnan(r):
                rhos_boot.append(r)
        ci_low = float(np.percentile(rhos_boot, 2.5)) if rhos_boot else float("nan")
        ci_high = float(np.percentile(rhos_boot, 97.5)) if rhos_boot else float("nan")

        h2_pass = (rho > 0) and (p_val_h2 < 0.05)
    else:
        rho, p_val_h2 = float("nan"), float("nan")
        ci_low, ci_high = float("nan"), float("nan")
        h2_pass = False

    g2_pass = h1a_pass

    decision = {
        "decision": "GO" if g2_pass else "NO-GO",
        "h1a": {
            "pass": bool(h1a_pass),
            "families_with_improvement": h1a_pass_count,
            "total_families": len(families),
            "details": h1a_details,
        },
        "h2": {
            "pass": bool(h2_pass),
            "spearman_rho": float(rho) if not np.isnan(rho) else None,
            "p_value": float(p_val_h2) if not np.isnan(p_val_h2) else None,
            "bootstrap_ci_95": [ci_low, ci_high],
            "optimal_tknee": {k: float(v) for k, v in optimal_tknee.items()},
            "spectral_gap_ratios": {k: float(v) for k, v in spectral_gaps.items()},
        },
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "g2_decision.json", "w") as f:
        json.dump(decision, f, indent=2)

    logger.info(f"G2 Decision: {decision['decision']}")
    return decision
