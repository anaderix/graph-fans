"""Compare 3L GCN vs 6L GCN vs 6L TransformerConv on generation quality."""

import json
import logging

import numpy as np

from graph_fans.phase0.spectral_profiler import compute_laplacian_spectrum, partition_into_bands
from graph_fans.phase2.dataset import load_dataset
from graph_fans.phase2.evaluate import _get_graph
from graph_fans.phase2.spectral_wasserstein import spectral_w1_summary
from graph_fans.phase2.trainer import Trainer, TrainConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    ds = load_dataset("results/phase2f_small/datasets/SBM_q0.01_seed0.npz")
    graph = _get_graph("SBM(q=0.01)", 50, seed=0)
    evals, evecs = compute_laplacian_spectrum(graph)
    _, bands = partition_into_bands(evals, B=8)

    configs = [
        ("3L GCN", dict(n_layers=3, conv_type="gcn")),
        ("6L GCN", dict(n_layers=6, conv_type="gcn")),
        ("6L Transformer", dict(n_layers=6, conv_type="transformer")),
    ]

    results = []
    for name, overrides in configs:
        print(f"\n=== {name} ===")
        cfg = TrainConfig(
            n_epochs=500, batch_timesteps=32, seed=0, device="cuda",
            hidden_dim=128, sde_type="cosine", use_ema=True,
            use_lr_scheduler=True, n_train_samples=100, **overrides,
        )
        trainer = Trainer(cfg, graph, ds["train"])
        history = trainer.train()
        sanity = trainer.sanity_check(ds["train"], n_gen=10)
        gen = trainer.generate(n_samples=50)
        w1 = spectral_w1_summary(ds["ref"], gen, evecs, bands)

        print(f"  final_loss={history['loss'][-1]:.4f}")
        print(f"  gen_std={sanity['gen_std']:.3f}, train_std={sanity['train_std']:.3f}, ratio={sanity['std_ratio']:.2f}")
        print(f"  spectral_L2={sanity['spectral_l2']:.3f}")
        print(f"  W1 total={w1['total_w1']:.1f}, low_band={w1['low_band_w1']:.1f}, high_band={w1['high_band_w1']:.1f}")
        print(f"  per_band_w1={[f'{v:.1f}' for v in w1['per_band_w1']]}")
        print(f"  warnings: {sanity['warnings']}")

        results.append({
            "name": name,
            "final_loss": history["loss"][-1],
            "gen_std": sanity["gen_std"],
            "train_std": sanity["train_std"],
            "std_ratio": sanity["std_ratio"],
            "spectral_l2": sanity["spectral_l2"],
            "w1_total": w1["total_w1"],
            "w1_low_band": w1["low_band_w1"],
            "w1_high_band": w1["high_band_w1"],
            "per_band_w1": w1["per_band_w1"].tolist(),
            "warnings": sanity["warnings"],
        })

    # Oracle baseline
    w1_oracle = spectral_w1_summary(ds["ref"], ds["train"][:50], evecs, bands)
    print(f"\nOracle W1: total={w1_oracle['total_w1']:.1f}, low={w1_oracle['low_band_w1']:.1f}, high={w1_oracle['high_band_w1']:.1f}")

    # Noise baseline
    rng = np.random.RandomState(42)
    noise = rng.randn(50, 50, 4)
    w1_noise = spectral_w1_summary(ds["ref"], noise, evecs, bands)
    print(f"Noise W1: total={w1_noise['total_w1']:.1f}, low={w1_noise['low_band_w1']:.1f}, high={w1_noise['high_band_w1']:.1f}")

    results.append({"name": "oracle", "w1_total": w1_oracle["total_w1"]})
    results.append({"name": "noise", "w1_total": w1_noise["total_w1"]})

    with open("results/diagnostics/arch_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/diagnostics/arch_comparison.json")


if __name__ == "__main__":
    main()
