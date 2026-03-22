"""Main script for Phase 0: Empirical Spectral Profiling.

Usage:
    uv run python -m graph_fans.phase0.run_profiling [--output-dir results/phase0] [--n-nodes 200] [--bands 8] [--seeds 3]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from graph_fans.utils.graph_generators import get_all_graph_families
from .spectral_profiler import spectral_energy_profile, SpectralProfile
from .visualize import plot_all_profiles, plot_energy_decay, plot_g0_summary

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_profiling(
    output_dir: str = "results/phase0",
    n_nodes: int = 200,
    B: int = 8,
    n_seeds: int = 3,
    n_features: int = 16,
) -> dict[str, SpectralProfile]:
    """Run spectral profiling on all graph families.

    Returns:
        Dictionary mapping graph name to averaged SpectralProfile.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_profiles: dict[str, list[SpectralProfile]] = {}

    for seed in range(n_seeds):
        logger.info(f"\n--- Seed {seed + 1}/{n_seeds} ---")
        graphs = get_all_graph_families(
            n_nodes=n_nodes, n_features=n_features, seed=seed
        )

        for gd in graphs:
            profile = spectral_energy_profile(
                gd.graph, gd.features, B=B, name=gd.name
            )
            all_profiles.setdefault(gd.name, []).append(profile)

    # Average profiles across seeds
    averaged: dict[str, SpectralProfile] = {}
    for name, profiles in all_profiles.items():
        energies = np.stack([p.band_energies for p in profiles])
        mean_energy = energies.mean(axis=0)
        std_energy = energies.std(axis=0)

        ratios = [p.energy_ratio for p in profiles]
        mean_ratio = np.mean(ratios)

        # Use the first seed's structural data (eigenvalues, eigenvectors, boundaries)
        base = profiles[0]
        total = mean_energy.sum()
        norm = mean_energy / total if total > 0 else mean_energy

        from .spectral_profiler import fit_decay_model

        decay_model, decay_params, decay_r2 = fit_decay_model(mean_energy)

        averaged[name] = SpectralProfile(
            name=name,
            eigenvalues=base.eigenvalues,
            eigenvectors=base.eigenvectors,
            band_boundaries=base.band_boundaries,
            band_energies=mean_energy,
            normalized_band_energies=norm,
            band_indices=base.band_indices,
            energy_ratio=mean_ratio,
            decay_model=decay_model,
            decay_params=decay_params,
            decay_r_squared=decay_r2,
        )

    # --- Generate outputs ---
    logger.info("\n=== Results Summary ===")

    # Summary table
    rows = []
    for name, p in averaged.items():
        rows.append(
            {
                "Graph Family": name,
                "Energy Ratio (max/min)": f"{p.energy_ratio:.2f}",
                "Decay Model": p.decay_model,
                "Decay R²": f"{p.decay_r_squared:.3f}",
                "Decay Params": str(p.decay_params),
                "G0 Pass": "YES" if p.energy_ratio >= 2.0 else "NO",
            }
        )
    df = pd.DataFrame(rows)
    logger.info(f"\n{df.to_string(index=False)}")
    df.to_csv(output_path / "summary.csv", index=False)

    # G0 decision
    passing = sum(1 for p in averaged.values() if p.energy_ratio >= 2.0)
    total_families = len(averaged)
    g0_pass = passing >= 2
    g0_result = {
        "decision": "GO" if g0_pass else "NO-GO",
        "passing_families": passing,
        "total_families": total_families,
        "threshold": "≥2× energy ratio in ≥2 families",
        "families": {
            name: {
                "energy_ratio": float(p.energy_ratio),
                "decay_model": p.decay_model,
                "decay_r_squared": float(p.decay_r_squared),
                "pass": bool(p.energy_ratio >= 2.0),
            }
            for name, p in averaged.items()
        },
    }
    with open(output_path / "g0_decision.json", "w") as f:
        json.dump(g0_result, f, indent=2)

    logger.info(
        f"\nG0 Decision: {'GO' if g0_pass else 'NO-GO'} "
        f"({passing}/{total_families} families pass)"
    )

    # Plots
    logger.info("\nGenerating plots...")
    plot_all_profiles(averaged, save_path=output_path / "spectral_profiles")
    plot_energy_decay(averaged, save_path=output_path / "energy_decay")
    plot_g0_summary(averaged, save_path=output_path / "g0_summary")

    # Save raw band energies for Phase 1a
    energy_data = {
        name: {
            "band_energies": p.band_energies.tolist(),
            "normalized_band_energies": p.normalized_band_energies.tolist(),
            "band_boundaries": [(float(lo), float(hi)) for lo, hi in p.band_boundaries],
            "energy_ratio": float(p.energy_ratio),
        }
        for name, p in averaged.items()
    }
    with open(output_path / "band_energies.json", "w") as f:
        json.dump(energy_data, f, indent=2)

    logger.info(f"\nResults saved to {output_path}/")
    return averaged


def main():
    parser = argparse.ArgumentParser(description="Phase 0: Empirical Spectral Profiling")
    parser.add_argument("--output-dir", default="results/phase0", help="Output directory")
    parser.add_argument("--n-nodes", type=int, default=200, help="Number of nodes for synthetic graphs")
    parser.add_argument("--bands", type=int, default=8, help="Number of spectral bands")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--n-features", type=int, default=16, help="Number of node features")
    args = parser.parse_args()

    run_profiling(
        output_dir=args.output_dir,
        n_nodes=args.n_nodes,
        B=args.bands,
        n_seeds=args.seeds,
        n_features=args.n_features,
    )


if __name__ == "__main__":
    main()
