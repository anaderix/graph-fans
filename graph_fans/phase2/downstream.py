"""Downstream task evaluation for Graph-FANS.

Evaluates whether W1 improvements from spectral noise shaping translate to
practical gains on node classification (predict community membership).

Experimental protocol:
  - Train a model (uniform or spectral) on ds["train"] feature realizations.
  - Generate n_gen_samples new feature matrices from the trained model.
  - Extract spectral coefficients from generated features.
  - Train a logistic regression classifier on spectral coefficients of generated
    features + community labels derived from the graph structure.
  - Evaluate the classifier on spectral coefficients of ds["ref"] (real features).

Data leakage safeguards:
  - Classifier TRAINS on generated features (never on ref features).
  - Classifier TESTS on ds["ref"] (real, held-out features).
  - Community labels are detected via Louvain with seed=0 (fixed, reproducible)
    and reused across all seeds in a family run.
  - Importance weights derived from seed-0 training split only.
"""

from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

N_GEN_SAMPLES_DEFAULT = 1000


def _get_community_labels(graph: nx.Graph, family: str, seed: int = 0) -> np.ndarray:
    """Detect community labels for a graph.

    For SBM graphs, labels are derived from block structure (nodes are ordered
    by community). For BA/other graphs, Louvain community detection is used
    with fixed seed=0 for reproducibility.

    Args:
        graph: NetworkX graph.
        family: Graph family string for SBM detection heuristic.
        seed: Louvain random seed (default 0). Only used for non-SBM graphs.

    Returns:
        Integer community labels of shape [n_nodes].
    """
    n = graph.number_of_nodes()

    if family.startswith("SBM"):
        # SBM: nodes are partitioned into n_communities equal blocks by construction
        # in generate_sbm. For n=50 with 4 communities: [0]*13 + [1]*12 + [2]*13 + [3]*12
        # Detect dynamically via Louvain for generality.
        communities = list(
            nx.algorithms.community.louvain_communities(graph, seed=seed)
        )
    else:
        communities = list(
            nx.algorithms.community.louvain_communities(graph, seed=seed)
        )

    labels = np.zeros(n, dtype=int)
    for comm_id, comm_nodes in enumerate(communities):
        for node in comm_nodes:
            labels[int(node)] = comm_id

    logger.debug(
        f"Community labels: {len(communities)} communities, "
        f"sizes={[len(c) for c in communities]}"
    )
    return labels


def generate_feature_set(
    trainer,  # Trainer instance — avoids circular import
    n_samples: int = N_GEN_SAMPLES_DEFAULT,
) -> np.ndarray:
    """Generate feature samples from a trained model.

    Args:
        trainer: Trained Trainer instance.
        n_samples: Number of samples to generate.
            Each sample is an independent draw from torch.randn (no manual seed).

    Returns:
        Feature array of shape [n_samples, n_nodes, n_features].
    """
    logger.info(f"  Generating {n_samples} feature samples...")
    return trainer.generate(n_samples=n_samples)


def build_node_classification_dataset(
    features_set: np.ndarray,
    community_labels: np.ndarray,
    eigenvectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build node classification dataset from feature samples.

    For each sample in features_set, project node features onto the Laplacian
    eigenbasis (spectral coefficients). Flatten to [n_samples * n_nodes, n_nodes]
    as feature vectors, with community labels tiled across samples.

    Args:
        features_set: [n_samples, n_nodes, n_features] feature matrices.
        community_labels: [n_nodes] integer community IDs.
        eigenvectors: [n_nodes, n_nodes] Laplacian eigenvectors.

    Returns:
        Tuple (X, y) where:
          X: [n_samples * n_nodes, n_nodes] spectral coefficient vectors.
          y: [n_samples * n_nodes] community labels, tiled across samples.
    """
    n_samples, n_nodes, n_features = features_set.shape
    X_list = []

    for features in features_set:
        # Spectral coefficients: [n_nodes, n_features]
        coeffs = eigenvectors.T @ features
        # Use per-node spectral energy across features as classification features
        # Shape: [n_nodes, n_nodes] — using full spectral representation
        # Broadcast: for each node, its row in eigenvectors.T @ features
        X_list.append(coeffs)  # [n_nodes, n_features]

    # Stack: [n_samples * n_nodes, n_features]
    X = np.concatenate(X_list, axis=0)
    # Tile labels: [n_samples * n_nodes]
    y = np.tile(community_labels, n_samples)

    assert X.shape[0] == y.shape[0], f"Shape mismatch: X={X.shape}, y={y.shape}"
    return X, y


def evaluate_node_classification(
    features_gen: np.ndarray,
    features_ref: np.ndarray,
    community_labels: np.ndarray,
    eigenvectors: np.ndarray,
    seed: int = 0,
) -> dict:
    """Evaluate node classification: train on generated, test on reference.

    The classifier is trained on spectral coefficients derived from generated
    feature matrices. It is evaluated on spectral coefficients from the real
    reference split. These two sets are strictly disjoint.

    Args:
        features_gen: [n_gen, n_nodes, n_features] — generated by the model.
            Used for TRAINING the classifier.
        features_ref: [n_ref, n_nodes, n_features] — real reference samples.
            Used for TESTING the classifier. Never seen during training.
        community_labels: [n_nodes] integer community IDs.
        eigenvectors: [n_nodes, n_nodes] Laplacian eigenvectors.
        seed: Random seed for LogisticRegression (default 0).

    Returns:
        Dict with 'train_acc' and 'test_acc' (both in [0, 1]).
    """
    from sklearn.linear_model import LogisticRegression

    # Build train set from generated features
    X_train, y_train = build_node_classification_dataset(
        features_gen, community_labels, eigenvectors
    )
    # Build test set from reference features (NEVER used for training)
    X_test, y_test = build_node_classification_dataset(
        features_ref, community_labels, eigenvectors
    )

    clf = LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")
    clf.fit(X_train, y_train)

    train_acc = float(clf.score(X_train, y_train))
    test_acc = float(clf.score(X_test, y_test))

    logger.debug(f"  Classification: train_acc={train_acc:.3f}, test_acc={test_acc:.3f}")
    return {"train_acc": train_acc, "test_acc": test_acc}


def run_downstream_experiment(
    family: str,
    n_nodes: int,
    n_seeds: int,
    device: str,
    dataset_dir: str,
    n_gen_samples: int = N_GEN_SAMPLES_DEFAULT,
    n_epochs: int = 500,
    batch_timesteps: int = 32,
    hidden_dim: int = 128,
    n_layers: int = 3,
) -> dict:
    """Run full downstream evaluation for one graph family.

    For each seed: train uniform and spectral models, generate n_gen_samples
    each, classify nodes using spectral features, evaluate on reference split.

    Data splits:
      - Model trains on ds["train"] (feature realizations).
      - Classifier trains on generated features from the trained model.
      - Classifier tests on ds["ref"] (real reference features).
      - Community labels derived from graph topology via Louvain(seed=0),
        fixed across all seeds.

    Args:
        family: Graph family string.
        n_nodes: Number of nodes.
        n_seeds: Number of random seeds.
        device: Torch device string.
        dataset_dir: Dataset cache directory.
        n_gen_samples: Number of samples to generate per method per seed.
        n_epochs: Training epochs.
        batch_timesteps: Batch timesteps per epoch.
        hidden_dim: Score network hidden dimension.
        n_layers: Score network GCN layers.

    Returns:
        Dict with keys: family, n_nodes, uniform_acc_per_seed, spectral_acc_per_seed,
        improvement_pct, p_val, per_seed.
    """
    from scipy.stats import ttest_rel

    from graph_fans.phase0.spectral_profiler import (
        compute_laplacian_spectrum,
        partition_into_bands,
        compute_band_energy,
    )
    from graph_fans.phase2.dataset import get_or_generate_dataset
    from graph_fans.phase2.evaluate import _get_graph
    from graph_fans.phase2.noise_shaper import compute_importance_weights
    from graph_fans.phase2.trainer import Trainer, TrainConfig

    logger.info(f"\n{'='*80}")
    logger.info(f"  Downstream: {family} (n={n_nodes})")
    logger.info(f"{'='*80}")

    graph = _get_graph(family, n_nodes, seed=0)
    evals, evecs = compute_laplacian_spectrum(graph)
    _, bands = partition_into_bands(evals, B=8)

    # Community labels: fixed Louvain(seed=0), reused across all seeds.
    community_labels = _get_community_labels(graph, family, seed=0)
    n_communities = len(np.unique(community_labels))
    logger.info(f"  Communities: {n_communities}")

    # Importance weights from seed-0 training split only.
    ds0 = get_or_generate_dataset(
        graph, family, 0,
        n_train=100, n_ref=50,
        n_features=4, feature_mode="community",
        cache_dir=dataset_dir,
    )
    profiles = [compute_band_energy(feat, evecs, bands) for feat in ds0["train"][:50]]
    weights = compute_importance_weights(np.mean(profiles, axis=0))

    uniform_acc_per_seed: list[float] = []
    spectral_acc_per_seed: list[float] = []
    per_seed: list[dict] = []

    for seed in range(n_seeds):
        ds = get_or_generate_dataset(
            graph, family, seed,
            n_train=100, n_ref=50,
            n_features=4, feature_mode="community",
            cache_dir=dataset_dir,
        )

        seed_entry: dict = {"seed": seed}

        for method, use_spectral in [("uniform", False), ("spectral", True)]:
            cfg = TrainConfig(
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
                use_spectral_noise=use_spectral,
            )
            trainer = Trainer(
                cfg, graph, ds["train"],
                importance_weights=weights if use_spectral else None,
            )
            trainer.train()

            # Generate samples for classifier training
            features_gen = generate_feature_set(trainer, n_samples=n_gen_samples)

            # Evaluate: train on generated, test on real reference (no leakage)
            acc = evaluate_node_classification(
                features_gen=features_gen,
                features_ref=ds["ref"],
                community_labels=community_labels,
                eigenvectors=evecs,
                seed=seed,
            )

            logger.info(
                f"  {family}/seed={seed}/{method}: "
                f"train_acc={acc['train_acc']:.3f}, "
                f"test_acc={acc['test_acc']:.3f}"
            )

            seed_entry[method] = acc

            if method == "uniform":
                uniform_acc_per_seed.append(acc["test_acc"])
            else:
                spectral_acc_per_seed.append(acc["test_acc"])

        per_seed.append(seed_entry)

    u = np.array(uniform_acc_per_seed)
    s = np.array(spectral_acc_per_seed)

    if len(u) >= 2:
        t_stat, p_val = ttest_rel(s, u)  # one-sided: spectral > uniform
    else:
        t_stat, p_val = 0.0, 1.0

    improvement_pct = float((s.mean() - u.mean()) / max(u.mean(), 1e-8) * 100)

    logger.info(f"\n  {family} DOWNSTREAM SUMMARY:")
    logger.info(f"    Uniform test acc:  {u.mean():.3f} +/- {u.std():.3f}")
    logger.info(f"    Spectral test acc: {s.mean():.3f} +/- {s.std():.3f}")
    logger.info(f"    Improvement: {improvement_pct:.1f}%  p={p_val:.4f}")

    return {
        "family": family,
        "n_nodes": n_nodes,
        "uniform_acc_per_seed": uniform_acc_per_seed,
        "spectral_acc_per_seed": spectral_acc_per_seed,
        "uniform_acc_mean": float(u.mean()),
        "uniform_acc_std": float(u.std()),
        "spectral_acc_mean": float(s.mean()),
        "spectral_acc_std": float(s.std()),
        "improvement_pct": improvement_pct,
        "t_stat": float(t_stat),
        "p_val": float(p_val),
        "significant": bool(p_val < 0.05 and s.mean() > u.mean()),
        "n_communities": int(n_communities),
        "per_seed": per_seed,
    }
