"""Visualize synthetic graph structure, feature distributions, and spectra.

Generates and analyzes all graph families with all feature modes,
producing a comprehensive diagnostic report.

Usage:
    uv run python scripts/visualize_synthetic_graphs.py [--output-dir results/graph_diagnostics]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde

from graph_fans.utils.graph_generators import (
    generate_sbm, generate_ba, GraphData,
    FEATURE_SMOOTH, FEATURE_MULTISCALE, FEATURE_COMMUNITY,
)
from graph_fans.utils.multiscale_features import compute_structural_roles
from graph_fans.phase0.spectral_profiler import (
    compute_laplacian_spectrum, partition_into_bands, compute_band_energy,
)
from graph_fans.utils.save import save_figure

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)


def plot_graph_overview(gd: GraphData, roles: dict, ax_graph: plt.Axes, ax_degree: plt.Axes):
    """Plot graph layout colored by centrality + degree distribution."""
    G = gd.graph
    pos = nx.spring_layout(G, seed=42, k=1.5 / np.sqrt(G.number_of_nodes()))

    centrality = roles["centrality"]
    node_size = 20 + 80 * centrality
    nx.draw_networkx_edges(G, pos, ax=ax_graph, alpha=0.08, width=0.3)
    sc = nx.draw_networkx_nodes(
        G, pos, ax=ax_graph, node_size=node_size,
        node_color=centrality, cmap="viridis", vmin=0, vmax=1,
    )
    ax_graph.set_title(gd.name, fontsize=11)
    ax_graph.axis("off")

    # Degree distribution
    degrees = [d for _, d in G.degree()]
    ax_degree.hist(degrees, bins=20, color="steelblue", edgecolor="white", alpha=0.8, density=True)
    ax_degree.set_xlabel("Degree")
    ax_degree.set_ylabel("Density")
    ax_degree.set_title(f"Degree dist (mean={np.mean(degrees):.1f})")


def plot_feature_distributions(
    gd: GraphData, roles: dict, eigenvalues: np.ndarray, eigenvectors: np.ndarray,
    axes: list[plt.Axes],
):
    """Plot feature distributions: per-node histograms, per-role, spectral profile."""
    features = gd.features
    n, d = features.shape

    # (a) Feature magnitude distribution per structural role
    ax = axes[0]
    feat_norms = np.linalg.norm(features, axis=1)
    centrality = roles["centrality"]
    boundary = roles["boundary"]

    # Split into terciles by centrality
    terciles = np.percentile(centrality, [33, 66])
    masks = {
        "Peripheral": centrality < terciles[0],
        "Mid-centrality": (centrality >= terciles[0]) & (centrality < terciles[1]),
        "Hub": centrality >= terciles[1],
    }
    colors = ["#2196F3", "#FF9800", "#E91E63"]
    for (label, mask), color in zip(masks.items(), colors):
        if mask.sum() > 0:
            ax.hist(feat_norms[mask], bins=15, alpha=0.5, label=label, color=color, density=True)
    ax.set_xlabel("Feature L2 norm")
    ax.set_ylabel("Density")
    ax.set_title("Feature magnitude by role")
    ax.legend(fontsize=7)

    # (b) First 2 PCA components colored by centrality
    ax = axes[1]
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    proj = pca.fit_transform(features)
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=centrality, cmap="viridis",
                    s=8, alpha=0.7, edgecolors="none")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA colored by centrality")

    # (c) Per-band energy profile
    ax = axes[2]
    B = 8
    _, band_indices = partition_into_bands(eigenvalues, B)
    band_energies = compute_band_energy(features, eigenvectors, band_indices)
    total = band_energies.sum()
    norm_energies = band_energies / total if total > 0 else band_energies

    colors_bands = sns.color_palette("viridis", B)
    ax.bar(range(B), norm_energies, color=colors_bands, edgecolor="white")
    ax.set_xlabel("Band (low → high eigenvalue)")
    ax.set_ylabel("Normalized energy")
    ratio = norm_energies.max() / max(norm_energies[norm_energies > 0].min(), 1e-10)
    ax.set_title(f"Spectral energy (ratio={ratio:.1f}×)")

    # (d) Eigenvalue density
    ax = axes[3]
    ev = eigenvalues[eigenvalues > 1e-8]
    if len(ev) > 2:
        kde = gaussian_kde(ev, bw_method="silverman")
        grid = np.linspace(0, eigenvalues.max(), 200)
        ax.fill_between(grid, kde(grid), alpha=0.3, color="steelblue")
        ax.plot(grid, kde(grid), color="steelblue", lw=1.5)
    ax.set_xlabel("Eigenvalue λ")
    ax.set_ylabel("Density")
    spectral_gap = eigenvalues[1] / eigenvalues[-1] if eigenvalues[-1] > 0 else 0
    ax.set_title(f"Eigenvalue density (gap={spectral_gap:.3f})")


def plot_role_spectra(
    gd: GraphData, roles: dict, eigenvalues: np.ndarray, eigenvectors: np.ndarray,
    ax: plt.Axes,
):
    """Plot per-band energy contribution from different structural roles."""
    features = gd.features
    B = 8
    _, band_indices = partition_into_bands(eigenvalues, B)
    centrality = roles["centrality"]

    # Split nodes into role groups
    terciles = np.percentile(centrality, [33, 66])
    groups = {
        "Peripheral": centrality < terciles[0],
        "Mid": (centrality >= terciles[0]) & (centrality < terciles[1]),
        "Hub": centrality >= terciles[1],
    }

    # Compute per-group band energy
    coeffs = eigenvectors.T @ features  # [n, d]

    x = np.arange(B)
    width = 0.25
    colors = ["#2196F3", "#FF9800", "#E91E63"]

    for i, (label, mask) in enumerate(groups.items()):
        node_indices = np.where(mask)[0]
        group_energies = np.zeros(B)
        # Zero out features for nodes NOT in this group, then project
        masked_features = np.zeros_like(features)
        masked_features[node_indices] = features[node_indices]
        group_coeffs = eigenvectors.T @ masked_features
        for b, bi in enumerate(band_indices):
            if len(bi) > 0:
                group_energies[b] = (group_coeffs[bi] ** 2).sum()

        total = group_energies.sum()
        if total > 0:
            group_energies = group_energies / total
        ax.bar(x + i * width, group_energies, width, label=label, color=colors[i], alpha=0.8)

    ax.set_xlabel("Band (low → high eigenvalue)")
    ax.set_ylabel("Normalized energy")
    ax.set_title("Spectral energy by structural role")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"B{b}" for b in range(B)])
    ax.legend(fontsize=8)


def analyze_single(
    gd: GraphData, output_dir: Path, feature_mode: str,
):
    """Full analysis for one graph + feature mode."""
    logger.info(f"Analyzing {gd.name} / {feature_mode}")

    G = gd.graph
    features = gd.features
    roles = compute_structural_roles(G)
    eigenvalues, eigenvectors = compute_laplacian_spectrum(G)

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    # Row 1: Graph overview + degree dist + role distributions
    ax_graph = fig.add_subplot(gs[0, 0:2])
    ax_degree = fig.add_subplot(gs[0, 2])
    ax_roles = fig.add_subplot(gs[0, 3])

    plot_graph_overview(gd, roles, ax_graph, ax_degree)

    # Role distributions
    for label, key, color in [
        ("Centrality", "centrality", "#E91E63"),
        ("Boundary", "boundary", "#FF9800"),
        ("Locality", "locality", "#2196F3"),
    ]:
        ax_roles.hist(roles[key], bins=20, alpha=0.4, label=label, color=color, density=True)
    ax_roles.set_xlabel("Score")
    ax_roles.set_title("Structural role distributions")
    ax_roles.legend(fontsize=7)

    # Row 2: Feature distributions
    axes_feat = [fig.add_subplot(gs[1, i]) for i in range(4)]
    plot_feature_distributions(gd, roles, eigenvalues, eigenvectors, axes_feat)

    # Row 3: Role-dependent spectra + additional diagnostics
    ax_role_spec = fig.add_subplot(gs[2, 0:2])
    plot_role_spectra(gd, roles, eigenvalues, eigenvectors, ax_role_spec)

    # Feature correlation with graph distance
    ax_corr = fig.add_subplot(gs[2, 2])
    # Sample node pairs and compute feature similarity vs graph distance
    n = G.number_of_nodes()
    rng = np.random.RandomState(42)
    n_pairs = min(2000, n * (n - 1) // 2)
    pairs = rng.choice(n, size=(n_pairs, 2), replace=True)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]

    feat_sims = []
    graph_dists = []
    shortest_paths = dict(nx.all_pairs_shortest_path_length(G))
    for i, j in pairs[:500]:
        if j in shortest_paths.get(i, {}):
            feat_sim = np.dot(features[i], features[j]) / (
                np.linalg.norm(features[i]) * np.linalg.norm(features[j]) + 1e-10
            )
            feat_sims.append(feat_sim)
            graph_dists.append(shortest_paths[i][j])

    if feat_sims:
        ax_corr.scatter(graph_dists, feat_sims, s=3, alpha=0.3, c="steelblue")
        # Binned means
        dist_arr = np.array(graph_dists)
        sim_arr = np.array(feat_sims)
        for d in sorted(set(graph_dists)):
            mask = dist_arr == d
            if mask.sum() > 5:
                ax_corr.scatter([d], [sim_arr[mask].mean()], s=50, c="red", zorder=5, edgecolors="white")
        ax_corr.set_xlabel("Shortest path distance")
        ax_corr.set_ylabel("Feature cosine similarity")
        ax_corr.set_title("Feature sim vs graph distance")

    # Summary text
    ax_text = fig.add_subplot(gs[2, 3])
    ax_text.axis("off")
    B = 8
    _, band_indices = partition_into_bands(eigenvalues, B)
    band_energies = compute_band_energy(features, eigenvectors, band_indices)
    total = band_energies.sum()
    norm_e = band_energies / total if total > 0 else band_energies

    summary = (
        f"Graph: {gd.name}\n"
        f"Features: {feature_mode}\n"
        f"Nodes: {n}, Edges: {G.number_of_edges()}\n"
        f"Density: {nx.density(G):.4f}\n"
        f"Spectral gap: {eigenvalues[1]/eigenvalues[-1]:.4f}\n"
        f"Energy ratio: {norm_e.max()/max(norm_e[norm_e>0].min(), 1e-10):.1f}×\n"
        f"Top band: B{np.argmax(norm_e)} ({norm_e.max():.3f})\n"
        f"Low band (B0): {norm_e[0]:.3f}\n"
        f"High band (B7): {norm_e[-1]:.3f}\n"
        f"Feat dims: {features.shape[1]}\n"
        f"Feat mean norm: {np.linalg.norm(features, axis=1).mean():.2f}\n"
    )
    ax_text.text(0.05, 0.95, summary, transform=ax_text.transAxes,
                 fontsize=9, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle(f"{gd.name} — {feature_mode} features", fontsize=14, y=1.01)

    slug = gd.name.replace("(", "").replace(")", "").replace("=", "").replace(".", "")
    save_figure(fig, output_dir / f"{slug}_{feature_mode}", f"{gd.name} ({feature_mode})")
    plt.close(fig)

    return {
        "graph": gd.name,
        "feature_mode": feature_mode,
        "n_nodes": n,
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "spectral_gap": eigenvalues[1] / eigenvalues[-1] if eigenvalues[-1] > 0 else 0,
        "energy_ratio": norm_e.max() / max(norm_e[norm_e > 0].min(), 1e-10),
        "band_energies": norm_e.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize synthetic graph diagnostics")
    parser.add_argument("--output-dir", default="results/graph_diagnostics")
    parser.add_argument("--n-nodes", type=int, default=200)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate graphs
    graphs_configs = [
        ("SBM(q=0.01)", lambda mode: generate_sbm(n_nodes=args.n_nodes, p_inter=0.01, feature_mode=mode)),
        ("SBM(q=0.05)", lambda mode: generate_sbm(n_nodes=args.n_nodes, p_inter=0.05, feature_mode=mode)),
        ("SBM(q=0.1)", lambda mode: generate_sbm(n_nodes=args.n_nodes, p_inter=0.1, feature_mode=mode)),
        ("BA(m=2)", lambda mode: generate_ba(n_nodes=args.n_nodes, m=2, feature_mode=mode)),
        ("BA(m=5)", lambda mode: generate_ba(n_nodes=args.n_nodes, m=5, feature_mode=mode)),
    ]

    feature_modes = [FEATURE_SMOOTH, FEATURE_MULTISCALE, FEATURE_COMMUNITY]
    all_stats = []

    for name, gen_fn in graphs_configs:
        for mode in feature_modes:
            gd = gen_fn(mode)
            stats = analyze_single(gd, output_dir, mode)
            all_stats.append(stats)

    # Summary comparison figure
    logger.info("Generating summary comparison...")
    fig, axes = plt.subplots(len(graphs_configs), len(feature_modes),
                              figsize=(5 * len(feature_modes), 4 * len(graphs_configs)))

    for i, (name, gen_fn) in enumerate(graphs_configs):
        for j, mode in enumerate(feature_modes):
            ax = axes[i, j] if len(graphs_configs) > 1 else axes[j]
            gd = gen_fn(mode)
            eigenvalues, eigenvectors = compute_laplacian_spectrum(gd.graph)
            _, band_indices = partition_into_bands(eigenvalues, 8)
            be = compute_band_energy(gd.features, eigenvectors, band_indices)
            total = be.sum()
            norm = be / total if total > 0 else be

            colors = sns.color_palette("viridis", 8)
            ax.bar(range(8), norm, color=colors, edgecolor="white")
            ratio = norm.max() / max(norm[norm > 0].min(), 1e-10)
            ax.set_title(f"{name}\n{mode} ({ratio:.0f}×)", fontsize=9)
            if i == len(graphs_configs) - 1:
                ax.set_xlabel("Band")
            if j == 0:
                ax.set_ylabel("Energy")

    fig.suptitle("Spectral Energy Profiles: All Graphs × All Feature Modes", fontsize=13, y=1.01)
    fig.tight_layout()
    save_figure(fig, output_dir / "summary_comparison", "Spectral Profiles Summary")
    plt.close(fig)

    # Print summary table
    import pandas as pd
    df = pd.DataFrame(all_stats)
    df = df.drop(columns=["band_energies"])
    logger.info(f"\n{df.to_string(index=False)}")
    df.to_csv(output_dir / "diagnostics_summary.csv", index=False)

    logger.info(f"\nAll diagnostics saved to {output_dir}/")


if __name__ == "__main__":
    main()
