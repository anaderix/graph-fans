"""Definitive test: does spectral noise shaping improve W1 on generated samples?

Trains 3L GCN (best generator) with uniform vs spectral noise on SBM(q=0.01),
generates 50 samples each, compares W1. Repeats over 5 seeds for statistics.
"""

import json
import logging

import numpy as np
from scipy.stats import ttest_rel

from graph_fans.phase0.spectral_profiler import compute_laplacian_spectrum, partition_into_bands, compute_band_energy
from graph_fans.phase2.dataset import load_dataset, get_or_generate_dataset
from graph_fans.phase2.evaluate import _get_graph
from graph_fans.phase2.noise_shaper import compute_importance_weights
from graph_fans.phase2.spectral_wasserstein import spectral_w1_summary
from graph_fans.phase2.trainer import Trainer, TrainConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

N_SEEDS = 5
N_GEN = 50
FAMILIES = ["SBM(q=0.01)", "SBM(q=0.05)"]


def main():
    results = []

    for family in FAMILIES:
        print(f"\n{'='*80}")
        print(f"  {family}")
        print(f"{'='*80}")

        graph = _get_graph(family, 50, seed=0)
        evals, evecs = compute_laplacian_spectrum(graph)
        _, bands = partition_into_bands(evals, B=8)

        # Importance weights from training data
        ds0 = get_or_generate_dataset(graph, family, 0, n_train=100, n_ref=50,
                                       n_features=4, feature_mode="community",
                                       cache_dir="results/phase2f_small/datasets")
        profiles = []
        for feat in ds0["train"][:50]:
            e = compute_band_energy(feat, evecs, bands)
            profiles.append(e)
        weights = compute_importance_weights(np.mean(profiles, axis=0))

        uniform_w1s = []
        spectral_w1s = []

        for seed in range(N_SEEDS):
            ds = get_or_generate_dataset(graph, family, seed, n_train=100, n_ref=50,
                                          n_features=4, feature_mode="community",
                                          cache_dir="results/phase2f_small/datasets")

            for method, use_spectral in [("uniform", False), ("spectral", True)]:
                cfg = TrainConfig(
                    n_epochs=500, batch_timesteps=32, seed=seed, device="cuda",
                    hidden_dim=128, n_layers=3, conv_type="gcn",
                    sde_type="cosine", use_ema=True, use_lr_scheduler=True,
                    n_train_samples=100, use_spectral_noise=use_spectral,
                )
                trainer = Trainer(cfg, graph, ds["train"],
                                  importance_weights=weights if use_spectral else None)
                history = trainer.train()
                gen = trainer.generate(n_samples=N_GEN)
                w1 = spectral_w1_summary(ds["ref"], gen, evecs, bands)
                sanity = trainer.sanity_check(ds["train"], n_gen=5)

                print(f"  {family}/seed={seed}/{method}: "
                      f"loss={history['loss'][-1]:.3f}, "
                      f"std_ratio={sanity['std_ratio']:.2f}, "
                      f"W1={w1['total_w1']:.1f} "
                      f"(low={w1['low_band_w1']:.1f}, high={w1['high_band_w1']:.1f})")

                entry = {
                    "family": family, "seed": seed, "method": method,
                    "final_loss": history["loss"][-1],
                    "std_ratio": sanity["std_ratio"],
                    "w1_total": w1["total_w1"],
                    "w1_low": w1["low_band_w1"],
                    "w1_high": w1["high_band_w1"],
                    "per_band_w1": w1["per_band_w1"].tolist(),
                }
                results.append(entry)

                if method == "uniform":
                    uniform_w1s.append(w1["total_w1"])
                else:
                    spectral_w1s.append(w1["total_w1"])

        # Paired t-test
        u = np.array(uniform_w1s)
        s = np.array(spectral_w1s)
        diff = u - s
        t_stat, p_val = ttest_rel(u, s)
        print(f"\n  {family} SUMMARY:")
        print(f"    Uniform W1:  {u.mean():.1f} ± {u.std():.1f}")
        print(f"    Spectral W1: {s.mean():.1f} ± {s.std():.1f}")
        print(f"    Improvement: {diff.mean():.1f} ({diff.mean()/u.mean()*100:.1f}%)")
        print(f"    t={t_stat:.3f}, p={p_val:.4f}")
        print(f"    {'SIGNIFICANT' if p_val < 0.025 else 'not significant'} (Bonferroni α=0.025)")

    with open("results/diagnostics/shaping_w1_test.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/diagnostics/shaping_w1_test.json")


if __name__ == "__main__":
    main()
