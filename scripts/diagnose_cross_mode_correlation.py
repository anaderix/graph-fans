"""Cross-mode spectral correlation diagnostic for real and synthetic datasets.

Computes the correlation matrix of spectral coefficients across eigenmodes
to assess whether independent per-mode diffusion (Phase 5a) is viable.

If modes are approximately independent (low off-diagonal correlation),
Phase 5a's independent per-mode MLP denoising is well-motivated.
If strong block structure exists, autoregressive (5b) or attention (5c) is needed.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from graph_fans.phase0.spectral_profiler import compute_laplacian_spectrum
from graph_fans.utils.graph_generators import load_citation_network, generate_sbm
from graph_fans.utils.multiscale_features import generate_feature_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_cross_mode_correlation(
    features: np.ndarray,
    eigenvectors: np.ndarray,
) -> dict:
    """Compute cross-mode spectral coefficient correlation.

    Args:
        features: [n_nodes, n_features] or [N_samples, n_nodes, n_features]
        eigenvectors: [n_nodes, n_nodes]

    Returns:
        Dict with correlation matrix, energy profile, and summary stats.
    """
    if features.ndim == 2:
        features = features[np.newaxis]

    n_samples, n_nodes, n_features = features.shape
    n_modes = eigenvectors.shape[1]

    all_energies = []
    for sample in features:
        coeffs = eigenvectors.T @ sample  # [n_modes, n_features]
        energies = (coeffs ** 2).sum(axis=1)  # [n_modes]
        all_energies.append(energies)

    energy_matrix = np.stack(all_energies)  # [N_samples, n_modes]

    energy_profile = energy_matrix.mean(axis=0)

    if n_samples > 1:
        corr = np.corrcoef(energy_matrix.T)  # [n_modes, n_modes]
    else:
        coeffs = eigenvectors.T @ features[0]  # [n_modes, n_features]
        corr = np.corrcoef(coeffs)  # [n_modes, n_modes]

    corr = np.nan_to_num(corr, nan=0.0)

    mask = ~np.eye(n_modes, dtype=bool)
    off_diag = np.abs(corr[mask])
    mean_abs_corr = float(off_diag.mean())
    max_abs_corr = float(off_diag.max())
    frac_above_03 = float((off_diag > 0.3).mean())
    frac_above_05 = float((off_diag > 0.5).mean())

    block_sizes = []
    for block_end in [5, 10, 15, 20]:
        if block_end > n_modes:
            break
        block = np.abs(corr[:block_end, :block_end])
        block_mask = ~np.eye(block_end, dtype=bool)
        block_mean = float(block[block_mask].mean())
        block_sizes.append({"modes": f"0-{block_end-1}", "mean_abs_corr": block_mean})

    cross_block_5 = np.abs(corr[:5, 5:min(15, n_modes)])
    cross_block_mean = float(cross_block_5.mean()) if cross_block_5.size > 0 else 0.0

    return {
        "correlation_matrix": corr,
        "energy_profile": energy_profile,
        "n_modes": n_modes,
        "n_samples": n_samples,
        "n_features": n_features,
        "summary": {
            "mean_abs_off_diagonal": mean_abs_corr,
            "max_abs_off_diagonal": max_abs_corr,
            "frac_above_0.3": frac_above_03,
            "frac_above_0.5": frac_above_05,
            "block_structure": block_sizes,
            "cross_block_0_5_vs_5_15": cross_block_mean,
        },
    }


def plot_correlation(
    result: dict, title: str, output_path: Path
) -> None:
    """Plot correlation matrix and energy profile."""
    corr = result["correlation_matrix"]
    energy = result["energy_profile"]
    n = result["n_modes"]
    summary = result["summary"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    im = axes[0].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes[0].set_title("Mode Correlation Matrix")
    axes[0].set_xlabel("Eigenmode k")
    axes[0].set_ylabel("Eigenmode k'")
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    im2 = axes[1].imshow(np.abs(corr), cmap="hot_r", vmin=0, vmax=1, aspect="auto")
    axes[1].set_title(f"|Correlation| (mean off-diag: {summary['mean_abs_off_diagonal']:.3f})")
    axes[1].set_xlabel("Eigenmode k")
    axes[1].set_ylabel("Eigenmode k'")
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    axes[2].bar(range(n), energy / energy.sum(), alpha=0.7, color="steelblue")
    axes[2].set_title("Normalized Energy Profile")
    axes[2].set_xlabel("Eigenmode k")
    axes[2].set_ylabel("Fraction of total energy")
    axes[2].set_xlim(-0.5, min(n, 50) - 0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {output_path}")


def analyze_dataset(name: str, graph, features, output_dir: Path) -> dict:
    """Run full cross-mode analysis on one dataset."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  {name} (n={graph.number_of_nodes()}, features={features.shape})")
    logger.info(f"{'='*60}")

    eigenvalues, eigenvectors = compute_laplacian_spectrum(graph)
    result = compute_cross_mode_correlation(features, eigenvectors)
    s = result["summary"]

    logger.info(f"  Modes: {result['n_modes']}")
    logger.info(f"  Mean |corr| off-diagonal: {s['mean_abs_off_diagonal']:.4f}")
    logger.info(f"  Max  |corr| off-diagonal: {s['max_abs_off_diagonal']:.4f}")
    logger.info(f"  Fraction |corr| > 0.3:    {s['frac_above_0.3']:.4f}")
    logger.info(f"  Fraction |corr| > 0.5:    {s['frac_above_0.5']:.4f}")
    logger.info(f"  Cross-block (0-4 vs 5-14): {s['cross_block_0_5_vs_5_15']:.4f}")
    for b in s["block_structure"]:
        logger.info(f"  Block {b['modes']}: mean |corr| = {b['mean_abs_corr']:.4f}")

    plot_correlation(result, name, output_dir / f"{name.replace(' ', '_').replace('(', '').replace(')', '')}_correlation.png")

    return {
        "name": name,
        "n_nodes": graph.number_of_nodes(),
        "n_features": result["n_features"],
        "n_samples": result["n_samples"],
        "summary": s,
        "energy_profile": result["energy_profile"].tolist(),
        "eigenvalues": eigenvalues.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-mode correlation diagnostic")
    parser.add_argument("--output", default="results/diagnostics/cross_mode_correlation.json")
    parser.add_argument("--datasets", default="all",
                        help="Comma-separated: cora,citeseer,sbm005,sbm01,bam2 or 'all'")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_run = args.datasets.lower().split(",") if args.datasets != "all" else [
        "cora", "citeseer", "pubmed", "sbm005", "sbm01", "bam2"
    ]

    all_results = []

    if "cora" in datasets_to_run:
        gd = load_citation_network("Cora")
        all_results.append(analyze_dataset("Cora (real, 1433 features)", gd.graph, gd.features, output_dir))

        gd4 = load_citation_network("Cora", n_features=4)
        all_results.append(analyze_dataset("Cora (real, 4 features)", gd4.graph, gd4.features, output_dir))

    if "citeseer" in datasets_to_run:
        gd = load_citation_network("CiteSeer")
        all_results.append(analyze_dataset("CiteSeer (real, 3703 features)", gd.graph, gd.features, output_dir))

    if "pubmed" in datasets_to_run:
        gd = load_citation_network("PubMed")
        all_results.append(analyze_dataset(f"PubMed (real, {gd.features.shape[1]} features)", gd.graph, gd.features, output_dir))

    if "sbm005" in datasets_to_run:
        gd = generate_sbm(n_nodes=50, p_inter=0.05, seed=0, feature_mode="community")
        dataset = generate_feature_dataset(gd.graph, n_samples=100, n_features=4, base_seed=0, mode="community")
        all_results.append(analyze_dataset("SBM(q=0.05) synthetic (100 samples)", gd.graph, dataset, output_dir))

    if "sbm01" in datasets_to_run:
        gd = generate_sbm(n_nodes=50, p_inter=0.1, seed=0, feature_mode="community")
        dataset = generate_feature_dataset(gd.graph, n_samples=100, n_features=4, base_seed=0, mode="community")
        all_results.append(analyze_dataset("SBM(q=0.1) synthetic (100 samples)", gd.graph, dataset, output_dir))

    if "bam2" in datasets_to_run:
        from graph_fans.utils.graph_generators import generate_ba
        gd = generate_ba(n_nodes=50, m=2, seed=0, feature_mode="community")
        dataset = generate_feature_dataset(gd.graph, n_samples=100, n_features=4, base_seed=0, mode="community")
        all_results.append(analyze_dataset("BA(m=2) synthetic (100 samples)", gd.graph, dataset, output_dir))

    serializable = []
    for r in all_results:
        serializable.append(r)

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"\nSaved results to {output_path}")

    print(f"\n{'='*70}")
    print(f"{'Dataset':<40} {'Mean |corr|':>12} {'> 0.3':>8} {'> 0.5':>8} {'Cross-blk':>10}")
    print(f"{'-'*70}")
    for r in all_results:
        s = r["summary"]
        print(f"{r['name']:<40} {s['mean_abs_off_diagonal']:>12.4f} {s['frac_above_0.3']:>7.1%} {s['frac_above_0.5']:>7.1%} {s['cross_block_0_5_vs_5_15']:>10.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
