"""Explore whether persistent homology reveals meaningful spectral band structure.

For the graph families that showed positive results in Phase 2g
(SBM q=0.05, SBM q=0.1, BA m=2), compute:
1. Laplacian eigenvalue spectrum with gap analysis
2. Persistent homology under 3 distance metrics:
   - Shortest path (integer, expected degenerate)
   - Effective resistance (spectral, continuous)
   - Diffusion distance (heat kernel, continuous)
3. Compare band boundary suggestions from each approach

Produces comparison plots in results/graph_diagnostics/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from ripser import ripser

from graph_fans.utils.graph_generators import generate_sbm, generate_ba
from graph_fans.phase0.spectral_profiler import (
    compute_laplacian_spectrum,
    partition_into_bands,
    compute_band_energy,
)
from graph_fans.utils.multiscale_features import generate_community_boundary_features

FAMILIES = {
    "SBM(q=0.05)": lambda: generate_sbm(n_nodes=50, p_inter=0.05, seed=0, feature_mode="community"),
    "SBM(q=0.1)": lambda: generate_sbm(n_nodes=50, p_inter=0.1, seed=0, feature_mode="community"),
    "BA(m=2)": lambda: generate_ba(n_nodes=50, m=2, seed=0, feature_mode="community"),
}

OUT_DIR = "results/graph_diagnostics"


def effective_resistance_distance(eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    """Compute pairwise effective resistance from spectral decomposition.

    R_eff(i,j) = sum_{k>0} (1/lambda_k) * (u_k(i) - u_k(j))^2
    """
    n = len(eigenvalues)
    nonzero = eigenvalues > 1e-10
    inv_evals = np.zeros_like(eigenvalues)
    inv_evals[nonzero] = 1.0 / eigenvalues[nonzero]

    # Scaled eigenvectors: v_k(i) = u_k(i) / sqrt(lambda_k)
    scaled = eigenvectors[:, nonzero] * np.sqrt(inv_evals[nonzero])  # [n, k]

    # Pairwise L2 distance in scaled eigenvector space = sqrt(R_eff)
    # R_eff(i,j) = ||scaled[i] - scaled[j]||^2
    dist = np.zeros((n, n))
    for i in range(n):
        diff = scaled - scaled[i]
        dist[i] = np.sum(diff ** 2, axis=1)
    return dist  # this is R_eff, not sqrt(R_eff)


def diffusion_distance(eigenvalues: np.ndarray, eigenvectors: np.ndarray, t: float = 1.0) -> np.ndarray:
    """Compute pairwise diffusion distance at time t.

    D_t(i,j)^2 = sum_{k>0} exp(-2*t*lambda_k) * (u_k(i) - u_k(j))^2
    """
    n = len(eigenvalues)
    weights = np.exp(-2 * t * eigenvalues)
    weights[eigenvalues < 1e-10] = 0  # zero eigenvalue contributes nothing

    scaled = eigenvectors * np.sqrt(weights)  # [n, n]

    dist = np.zeros((n, n))
    for i in range(n):
        diff = scaled - scaled[i]
        dist[i] = np.sum(diff ** 2, axis=1)
    return dist


def compute_persistence(dist_matrix: np.ndarray, max_dim: int = 1) -> dict:
    """Compute persistent homology from a distance matrix."""
    result = ripser(dist_matrix, maxdim=max_dim, distance_matrix=True)
    return result["dgms"]


def find_eigenvalue_gaps(eigenvalues: np.ndarray, n_gaps: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Find the largest gaps in the eigenvalue spectrum.

    Returns (boundary_values, gap_sizes) sorted by position.
    """
    nonzero = eigenvalues[eigenvalues > 1e-10]
    if len(nonzero) < 2:
        return np.array([]), np.array([])

    gaps = np.diff(nonzero)
    top_indices = np.argsort(gaps)[-n_gaps:]
    top_indices = np.sort(top_indices)

    boundaries = [0.5 * (nonzero[i] + nonzero[i + 1]) for i in top_indices]
    gap_sizes = gaps[top_indices]

    return np.array(boundaries), gap_sizes


def boundaries_from_persistence(dgms: list[np.ndarray], eigenvalues: np.ndarray, dim: int = 0) -> np.ndarray:
    """Extract band boundaries from persistence diagram.

    Strategy: long-lived H0 features correspond to well-separated clusters
    at different scales. Their death times mark scale transitions.

    For diffusion/resistance-based distances, death times map more naturally
    to eigenvalue scales than shortest-path distances.
    """
    diag = dgms[dim]
    finite = diag[np.isfinite(diag[:, 1])]
    if len(finite) < 2:
        return np.array([])

    lifetimes = finite[:, 1] - finite[:, 0]
    # Keep features with above-median lifetime (topologically significant)
    median_lt = np.median(lifetimes)
    significant = finite[lifetimes > median_lt]

    if len(significant) < 2:
        return np.array([])

    # Death times of significant features = scale transitions
    death_times = np.sort(significant[:, 1])

    # Find gaps between consecutive death times
    if len(death_times) < 2:
        return np.array([])

    gaps = np.diff(death_times)
    n_bounds = min(7, len(gaps))
    if n_bounds == 0:
        return np.array([])

    top_gap_idx = np.argsort(gaps)[-n_bounds:]
    top_gap_idx = np.sort(top_gap_idx)

    # Boundaries at midpoints of gaps in death times
    raw_boundaries = [0.5 * (death_times[i] + death_times[i + 1]) for i in top_gap_idx]

    # Map distance-space boundaries to eigenvalue-space
    # For effective resistance: R_eff ~ 1/lambda, so lambda ~ 1/R
    # For diffusion distance at t=1: D^2 ~ exp(-2*lambda), so lambda ~ -0.5*ln(D^2)
    # Use simple linear mapping to eigenvalue range as first approximation
    raw = np.array(raw_boundaries)
    if raw.max() > raw.min():
        # Linear map from distance scale to eigenvalue scale
        lam_min = eigenvalues[eigenvalues > 1e-10].min()
        lam_max = eigenvalues[-1]
        mapped = lam_max - (raw - raw.min()) / (raw.max() - raw.min()) * (lam_max - lam_min)
        mapped = np.clip(mapped, lam_min, lam_max)
        return np.sort(mapped)

    return np.array([])


def partition_by_boundaries(eigenvalues: np.ndarray, boundaries: np.ndarray) -> list[np.ndarray]:
    """Partition eigenvalue indices using given boundary values."""
    all_bounds = np.concatenate([[0], np.sort(boundaries), [eigenvalues[-1] + 1e-6]])
    bands = []
    for i in range(len(all_bounds) - 1):
        lo, hi = all_bounds[i], all_bounds[i + 1]
        indices = np.where((eigenvalues >= lo) & (eigenvalues < hi))[0]
        bands.append(indices)
    return bands


def band_energy_profile(eigenvectors, band_indices_list, feat_samples):
    """Compute mean normalized energy profile under a given band scheme."""
    profiles = []
    for feat in feat_samples:
        coeffs = eigenvectors.T @ feat
        mode_energy = np.sum(coeffs ** 2, axis=1)
        energies = []
        for indices in band_indices_list:
            if len(indices) == 0:
                energies.append(0.0)
            else:
                energies.append(mode_energy[indices].sum())
        e = np.array(energies)
        total = e.sum()
        profiles.append(e / total if total > 0 else e)
    return np.mean(profiles, axis=0)


def plot_family(family_name: str, graph: nx.Graph, out_dir: str) -> dict:
    """Generate full diagnostic plot for one graph family."""
    n = graph.number_of_nodes()
    eigenvalues, eigenvectors = compute_laplacian_spectrum(graph)

    # --- Compute 3 distance metrics ---
    # 1. Shortest path
    sp = dict(nx.all_pairs_shortest_path_length(graph))
    dist_sp = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_sp[i, j] = sp[i].get(j, n)

    # 2. Effective resistance
    dist_eff = effective_resistance_distance(eigenvalues, eigenvectors)

    # 3. Diffusion distance (t=1.0)
    dist_diff = diffusion_distance(eigenvalues, eigenvectors, t=1.0)

    # --- Persistence for each ---
    dgms_sp = compute_persistence(dist_sp)
    dgms_eff = compute_persistence(np.sqrt(dist_eff + 1e-12))  # ripser expects metric
    dgms_diff = compute_persistence(np.sqrt(dist_diff + 1e-12))

    # --- Eigenvalue gaps ---
    gap_boundaries, gap_sizes = find_eigenvalue_gaps(eigenvalues, n_gaps=7)

    # --- Persistence-derived boundaries ---
    eff_boundaries = boundaries_from_persistence(dgms_eff, eigenvalues, dim=0)
    diff_boundaries = boundaries_from_persistence(dgms_diff, eigenvalues, dim=0)

    # --- Uniform boundaries ---
    _, uniform_bands = partition_into_bands(eigenvalues, B=8)
    lambda_max = eigenvalues[-1]
    uniform_boundaries = np.linspace(0, lambda_max, 9)[1:-1]

    # --- Feature samples ---
    feat_samples = [generate_community_boundary_features(graph, n_features=4, seed=s) for s in range(10)]

    # --- Energy profiles under different band schemes ---
    energy_uniform = band_energy_profile(eigenvectors, uniform_bands, feat_samples)
    energy_gap = band_energy_profile(
        eigenvectors, partition_by_boundaries(eigenvalues, gap_boundaries), feat_samples
    ) if len(gap_boundaries) > 0 else energy_uniform
    energy_eff = band_energy_profile(
        eigenvectors, partition_by_boundaries(eigenvalues, eff_boundaries), feat_samples
    ) if len(eff_boundaries) > 0 else energy_uniform
    energy_diff = band_energy_profile(
        eigenvectors, partition_by_boundaries(eigenvalues, diff_boundaries), feat_samples
    ) if len(diff_boundaries) > 0 else energy_uniform

    # --- Importance weight contrast ---
    def weight_contrast(profile):
        """Max/min ratio of importance weights derived from profile."""
        eps = 1e-3
        w = (profile + eps) ** (-1.0)
        w = w / w.mean()
        return w.max() / max(w.min(), 1e-6)

    # --- PLOTTING ---
    fig = plt.figure(figsize=(20, 18))
    fig.suptitle(
        f"{family_name} (n={n}): Persistence & Eigenvalue Gap Band Analysis",
        fontsize=15, fontweight="bold", y=0.98,
    )
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)

    # Row 0: Eigenvalue spectrum with all boundary types
    ax = fig.add_subplot(gs[0, :])
    ax.stem(range(len(eigenvalues)), eigenvalues, linefmt="C0-", markerfmt="C0o", basefmt=" ")
    for b in uniform_boundaries:
        ax.axhline(b, color="gray", ls="--", alpha=0.4, lw=0.8)
    for b in gap_boundaries:
        ax.axhline(b, color="red", ls="-", alpha=0.6, lw=1.2)
    if len(eff_boundaries) > 0:
        for b in eff_boundaries:
            ax.axhline(b, color="green", ls="-.", alpha=0.7, lw=1.2)
    if len(diff_boundaries) > 0:
        for b in diff_boundaries:
            ax.axhline(b, color="purple", ls=":", alpha=0.8, lw=1.5)
    legend_els = [
        Line2D([0], [0], color="gray", ls="--", label="Uniform"),
        Line2D([0], [0], color="red", ls="-", label="Eigenvalue-gap"),
        Line2D([0], [0], color="green", ls="-.", label="Eff. resistance persistence"),
        Line2D([0], [0], color="purple", ls=":", lw=1.5, label="Diffusion persistence"),
    ]
    ax.legend(handles=legend_els, loc="upper left", fontsize=9)
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("λ")
    ax.set_title("Laplacian Spectrum with Band Boundaries")

    # Row 1: Persistence diagrams (3 distance metrics)
    for col, (name_d, dgms, dist) in enumerate([
        ("Shortest Path", dgms_sp, dist_sp),
        ("Eff. Resistance", dgms_eff, dist_eff),
        ("Diffusion (t=1)", dgms_diff, dist_diff),
    ]):
        ax = fig.add_subplot(gs[1, col])
        h0 = dgms[0]
        h0f = h0[np.isfinite(h0[:, 1])]
        if len(h0f) > 0:
            lt = h0f[:, 1] - h0f[:, 0]
            sc = ax.scatter(h0f[:, 0], h0f[:, 1], c=lt, cmap="viridis", s=25, edgecolors="k", linewidths=0.3, label="H0")
            plt.colorbar(sc, ax=ax, label="Lifetime", shrink=0.7)

        if len(dgms) > 1 and len(dgms[1]) > 0:
            h1 = dgms[1]
            h1f = h1[np.isfinite(h1[:, 1])]
            if len(h1f) > 0:
                ax.scatter(h1f[:, 0], h1f[:, 1], c="red", s=15, marker="^", alpha=0.6, label="H1")

        all_finite = np.concatenate([d[np.isfinite(d[:, 1])] for d in dgms if len(d) > 0])
        if len(all_finite) > 0:
            dmax = all_finite[:, 1].max() * 1.1
            ax.plot([0, dmax], [0, dmax], "k--", alpha=0.2)
            ax.set_xlim(-0.02 * dmax, dmax)
            ax.set_ylim(-0.02 * dmax, dmax)

        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_title(f"Persistence: {name_d}")
        ax.legend(fontsize=8, loc="lower right")

        # Annotate: number of unique death times and lifetime CV
        n_unique_deaths = len(np.unique(h0f[:, 1].round(6))) if len(h0f) > 0 else 0
        cv = float(lt.std() / lt.mean()) if len(h0f) > 0 and lt.mean() > 0 else 0
        ax.annotate(
            f"{len(h0f)} H0 feats\n{n_unique_deaths} unique deaths\nCV={cv:.2f}",
            xy=(0.02, 0.98), xycoords="axes fraction", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

    # Row 2: H0 persistence barcodes side-by-side
    for col, (name_d, dgms) in enumerate([
        ("Shortest Path", dgms_sp),
        ("Eff. Resistance", dgms_eff),
        ("Diffusion (t=1)", dgms_diff),
    ]):
        ax = fig.add_subplot(gs[2, col])
        h0 = dgms[0]
        h0f = h0[np.isfinite(h0[:, 1])]
        if len(h0f) > 0:
            lt = h0f[:, 1] - h0f[:, 0]
            sorted_idx = np.argsort(-lt)
            n_show = min(25, len(sorted_idx))
            colors = plt.cm.viridis(lt[sorted_idx[:n_show]] / lt.max())
            for rank in range(n_show):
                idx = sorted_idx[rank]
                ax.barh(rank, lt[idx], left=h0f[idx, 0], height=0.8, color=colors[rank], alpha=0.8)
            ax.invert_yaxis()
        ax.set_xlabel("Filtration value")
        ax.set_ylabel("Feature rank (by lifetime)")
        ax.set_title(f"H0 Barcode: {name_d}")

    # Row 3 left: Energy profiles under different band schemes
    ax = fig.add_subplot(gs[3, 0])
    n_bands_max = max(len(energy_uniform), len(energy_gap), len(energy_eff), len(energy_diff))
    def pad(arr, n):
        return np.pad(arr, (0, n - len(arr)))
    eu = pad(energy_uniform, n_bands_max)
    eg = pad(energy_gap, n_bands_max)
    ee = pad(energy_eff, n_bands_max)
    ed = pad(energy_diff, n_bands_max)
    x = np.arange(n_bands_max)
    w = 0.2
    ax.bar(x - 1.5*w, eu, w, label="Uniform", color="gray", alpha=0.7)
    ax.bar(x - 0.5*w, eg, w, label="Eig-gap", color="red", alpha=0.7)
    ax.bar(x + 0.5*w, ee, w, label="Eff-res pers.", color="green", alpha=0.7)
    ax.bar(x + 1.5*w, ed, w, label="Diffusion pers.", color="purple", alpha=0.7)
    ax.set_xlabel("Band index")
    ax.set_ylabel("Normalized energy")
    ax.set_title("Energy Profile by Band Scheme")
    ax.legend(fontsize=7)

    # Row 3 middle: Band sizes
    ax = fig.add_subplot(gs[3, 1])
    su = [len(b) for b in uniform_bands]
    sg = [len(b) for b in partition_by_boundaries(eigenvalues, gap_boundaries)] if len(gap_boundaries) > 0 else su
    se = [len(b) for b in partition_by_boundaries(eigenvalues, eff_boundaries)] if len(eff_boundaries) > 0 else su
    sd = [len(b) for b in partition_by_boundaries(eigenvalues, diff_boundaries)] if len(diff_boundaries) > 0 else su
    n_b = max(len(su), len(sg), len(se), len(sd))
    while len(su) < n_b: su.append(0)
    while len(sg) < n_b: sg.append(0)
    while len(se) < n_b: se.append(0)
    while len(sd) < n_b: sd.append(0)
    x = np.arange(n_b)
    ax.bar(x - 1.5*w, su, w, label="Uniform", color="gray", alpha=0.7)
    ax.bar(x - 0.5*w, sg, w, label="Eig-gap", color="red", alpha=0.7)
    ax.bar(x + 0.5*w, se, w, label="Eff-res pers.", color="green", alpha=0.7)
    ax.bar(x + 1.5*w, sd, w, label="Diffusion pers.", color="purple", alpha=0.7)
    ax.set_xlabel("Band index")
    ax.set_ylabel("# eigenmodes")
    ax.set_title("Band Sizes")
    ax.legend(fontsize=7)

    # Row 3 right: Eigenvalue gaps
    ax = fig.add_subplot(gs[3, 2])
    nonzero = eigenvalues[eigenvalues > 1e-10]
    if len(nonzero) > 1:
        gaps = np.diff(nonzero)
        ax.bar(range(len(gaps)), gaps, color="C1", alpha=0.6)
        top7 = np.argsort(gaps)[-7:]
        ax.bar(top7, gaps[top7], color="red", alpha=0.9)
    ax.set_xlabel("Gap index")
    ax.set_ylabel("Gap size (Δλ)")
    ax.set_title("Eigenvalue Gaps (top 7 in red)")

    # Save
    safe_name = family_name.replace("(", "").replace(")", "").replace("=", "").replace(".", "")
    path = f"{out_dir}/persistence_{safe_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # --- Collect stats ---
    def persistence_stats(dgms, name):
        h0 = dgms[0]
        h0f = h0[np.isfinite(h0[:, 1])]
        lt = h0f[:, 1] - h0f[:, 0] if len(h0f) > 0 else np.array([0])
        cv = float(lt.std() / lt.mean()) if lt.mean() > 0 else 0.0
        n_unique_deaths = len(np.unique(h0f[:, 1].round(6))) if len(h0f) > 0 else 0
        return {
            "distance": name,
            "n_h0": len(h0f),
            "n_unique_h0_deaths": n_unique_deaths,
            "h0_lifetime_cv": round(cv, 4),
            "h0_max_lifetime": round(float(lt.max()), 4) if len(lt) > 0 else 0,
            "h0_lifetime_range": [round(float(lt.min()), 4), round(float(lt.max()), 4)] if len(lt) > 0 else [0, 0],
            "n_h1": len(dgms[1]) if len(dgms) > 1 else 0,
        }

    return {
        "family": family_name,
        "n_nodes": n,
        "lambda_max": round(float(eigenvalues[-1]), 4),
        "spectral_gap": round(float(eigenvalues[1]), 4) if len(eigenvalues) > 1 else 0,
        "persistence": {
            "shortest_path": persistence_stats(dgms_sp, "shortest_path"),
            "effective_resistance": persistence_stats(dgms_eff, "effective_resistance"),
            "diffusion_t1": persistence_stats(dgms_diff, "diffusion_t1"),
        },
        "n_gap_boundaries": len(gap_boundaries),
        "n_eff_boundaries": len(eff_boundaries),
        "n_diff_boundaries": len(diff_boundaries),
        "gap_boundaries": [round(b, 4) for b in gap_boundaries],
        "eff_boundaries": [round(b, 4) for b in eff_boundaries],
        "diff_boundaries": [round(b, 4) for b in diff_boundaries],
        "uniform_boundaries": [round(b, 4) for b in uniform_boundaries],
        "weight_contrast": {
            "uniform": round(weight_contrast(energy_uniform), 2),
            "eigenvalue_gap": round(weight_contrast(energy_gap), 2),
            "eff_resistance": round(weight_contrast(energy_eff), 2) if len(eff_boundaries) > 0 else None,
            "diffusion": round(weight_contrast(energy_diff), 2) if len(diff_boundaries) > 0 else None,
        },
    }


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    all_stats = []

    for name, gen_fn in FAMILIES.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        gd = gen_fn()
        stats = plot_family(name, gd.graph, OUT_DIR)
        all_stats.append(stats)

        # Summary
        print(f"  λ_max={stats['lambda_max']}, spectral gap={stats['spectral_gap']}")
        for dname, pstats in stats["persistence"].items():
            print(f"  {dname}: {pstats['n_h0']} H0 features, "
                  f"{pstats['n_unique_h0_deaths']} unique deaths, "
                  f"lifetime CV={pstats['h0_lifetime_cv']:.3f}, "
                  f"range={pstats['h0_lifetime_range']}")
        print(f"  Boundaries: gap={len(stats['gap_boundaries'])}, "
              f"eff-res={len(stats['eff_boundaries'])}, "
              f"diffusion={len(stats['diff_boundaries'])}")
        wc = stats["weight_contrast"]
        print(f"  Weight contrast: uniform={wc['uniform']}, gap={wc['eigenvalue_gap']}, "
              f"eff-res={wc['eff_resistance']}, diff={wc['diffusion']}")

    with open(f"{OUT_DIR}/persistence_analysis.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved to {OUT_DIR}/persistence_analysis.json")


if __name__ == "__main__":
    main()
