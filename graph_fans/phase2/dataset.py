"""Dataset persistence: pre-generate, save, load, and validate feature datasets.

Eliminates on-the-fly feature generation, enabling inspection before training
and consistent caching across experiment runs.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np

from graph_fans.phase0.spectral_profiler import (
    compute_laplacian_spectrum,
    partition_into_bands,
    compute_band_energy,
)
from graph_fans.utils.multiscale_features import generate_feature_dataset

logger = logging.getLogger(__name__)


def generate_and_save_dataset(
    graph: nx.Graph,
    family: str,
    seed: int,
    n_train: int = 500,
    n_ref: int = 50,
    n_features: int = 16,
    feature_mode: str = "community",
    output_dir: str | Path = "results/phase2/datasets",
) -> Path:
    """Generate train + reference datasets and save as .npz.

    Args:
        graph: Graph topology.
        family: Graph family name (e.g. "SBM(q=0.05)").
        seed: Experiment seed.
        n_train: Number of training feature realizations.
        n_ref: Number of held-out reference realizations.
        n_features: Feature dimensionality.
        feature_mode: Feature generation mode.
        output_dir: Directory to save the .npz file.

    Returns:
        Path to the saved .npz file.
    """
    train_base_seed = seed * 10000
    ref_base_seed = seed * 10000 + 100000

    logger.info(f"  Generating {n_train} train + {n_ref} ref samples for {family} seed={seed}...")
    train_data = generate_feature_dataset(
        graph, n_samples=n_train, n_features=n_features,
        base_seed=train_base_seed, mode=feature_mode,
    )
    ref_data = generate_feature_dataset(
        graph, n_samples=n_ref, n_features=n_features,
        base_seed=ref_base_seed, mode=feature_mode,
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Sanitize family name for filename
    safe_family = family.replace("(", "_").replace(")", "").replace("=", "")
    filename = f"{safe_family}_seed{seed}.npz"
    filepath = out_path / filename

    np.savez(
        filepath,
        train=train_data,
        ref=ref_data,
        family=family,
        seed=seed,
        n_features=n_features,
        feature_mode=feature_mode,
        created_at=datetime.now().isoformat(),
    )
    logger.info(f"  Saved dataset: {filepath} (train={train_data.shape}, ref={ref_data.shape})")
    return filepath


def load_dataset(path: str | Path) -> dict:
    """Load a dataset from .npz file.

    Returns:
        Dict with keys: 'train', 'ref', 'family', 'seed', 'n_features',
        'feature_mode', 'created_at'.
    """
    data = np.load(path, allow_pickle=True)
    return {
        "train": data["train"],
        "ref": data["ref"],
        "family": str(data["family"]),
        "seed": int(data["seed"]),
        "n_features": int(data["n_features"]),
        "feature_mode": str(data["feature_mode"]),
        "created_at": str(data["created_at"]),
    }


def validate_dataset(
    dataset: dict,
    graph: nx.Graph,
    B: int = 8,
) -> dict:
    """Validate dataset quality: check spectral profile and feature statistics.

    Returns:
        Dict with 'train_std', 'ref_std', 'spectral_profile', 'is_flat_spectrum',
        'warnings'.
    """
    train_data = dataset["train"]
    ref_data = dataset["ref"]

    train_std = float(train_data.std())
    ref_std = float(ref_data.std())

    eigenvalues, eigenvectors = compute_laplacian_spectrum(graph)
    _, band_indices = partition_into_bands(eigenvalues, B)

    # Mean spectral profile over training samples
    profiles = []
    for features in train_data[:min(50, len(train_data))]:
        energy = compute_band_energy(features, eigenvectors, band_indices)
        total = energy.sum()
        profiles.append(energy / total if total > 0 else energy)
    mean_profile = np.stack(profiles).mean(axis=0)

    # Check if spectrum is flat (all bands within 2× of each other)
    nonzero = mean_profile[mean_profile > 1e-6]
    is_flat = len(nonzero) > 0 and (nonzero.max() / nonzero.min()) < 2.0

    warnings = []
    if is_flat:
        warnings.append("Flat spectral profile — features may lack spectral structure")
    if train_std < 0.1:
        warnings.append(f"Very low train std={train_std:.4f}")
    if train_std > 10.0:
        warnings.append(f"Very high train std={train_std:.4f}")

    result = {
        "train_std": train_std,
        "ref_std": ref_std,
        "spectral_profile": mean_profile,
        "is_flat_spectrum": is_flat,
        "warnings": warnings,
    }

    for w in warnings:
        logger.warning(f"  Dataset validation: {w}")

    return result


def get_or_generate_dataset(
    graph: nx.Graph,
    family: str,
    seed: int,
    n_train: int = 500,
    n_ref: int = 50,
    n_features: int = 16,
    feature_mode: str = "community",
    cache_dir: str | Path = "results/phase2/datasets",
) -> dict:
    """Load dataset from cache or generate + save if not found.

    Returns:
        Dataset dict (same as load_dataset output).
    """
    cache_path = Path(cache_dir)
    safe_family = family.replace("(", "_").replace(")", "").replace("=", "")
    filename = f"{safe_family}_seed{seed}.npz"
    filepath = cache_path / filename

    if filepath.exists():
        logger.info(f"  Loading cached dataset: {filepath}")
        return load_dataset(filepath)

    generate_and_save_dataset(
        graph, family, seed, n_train, n_ref, n_features, feature_mode, cache_dir,
    )
    return load_dataset(filepath)
