import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


PRETTY_MODEL_NAMES = {
    "base_pretrained": "base_pretrained",
    "free": "free",
    "free_aug": "free_aug",
    "const_lam0.25": "const (λ=0.25)",
    "const_lam0.5": "const (λ=0.50)",
    "const_lam1.0": "const (λ=1.00)",
}

PRETTY_CKA_TITLES = {
    "free_s42 vs free_s123 (Controllo Seed)": "free (s42) vs. free (s123) [Seed Control]",
    "const05_s42 vs const05_s123 (Controllo Seed)": "const (λ=0.5, s42) vs. const (λ=0.5, s123) [Seed Control]",
    "free_s42 vs const05_s42 (Libera vs Vincolata 0.5)": "free (s42) vs. const (λ=0.5, s42)",
    "free_s42 vs const10_s42 (Libera vs Vincolata 1.0)": "free (s42) vs. const (λ=1.0, s42)",
}


def pretty_name(k: str) -> str:
    return PRETTY_MODEL_NAMES.get(k, k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = os.path.join(args.save_dir, "plots_en")
    os.makedirs(out_dir, exist_ok=True)

    rep_all_path = os.path.join(args.save_dir, "analysis_all_seed42", "analysis_report.json")
    rep_cs_path = os.path.join(args.save_dir, "analysis_cross_seed", "cross_seed_report.json")

    with open(rep_all_path) as f:
        rep_all = json.load(f)
    with open(rep_cs_path) as f:
        rep_cs = json.load(f)

    layers = list(range(7))

    # 1. Invariance Curve
    plt.figure(figsize=(6.8, 4.6))
    for name, curve in rep_all["invariance_curves_id"].items():
        plt.plot(layers, curve, marker="o", linewidth=2, label=pretty_name(name))
    plt.axvline(x=3, color="gray", linestyle="--", alpha=0.65, label="Constrained Layer (L3)")
    plt.title("Layer-wise Symbol Invariance (Test ID - Centered Cosine)", fontsize=11)
    plt.xlabel("Layer (0 = Embeddings, 1..6 = Transformer Layers)", fontsize=10)
    plt.ylabel("Centered Cosine Similarity (x vs. σ(x))", fontsize=10)
    plt.xticks(layers)
    plt.ylim(0.55, 1.02)
    plt.legend(fontsize=8.5, loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "invariance_curve.png"), dpi=220)
    plt.close()

    # 2. Probing Curves (Class ID + QDep OOD)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    for name, curve in rep_all["probe_cls_id"].items():
        axes[0].plot(layers, curve, marker="o", linewidth=2, label=pretty_name(name))
    axes[0].axvline(x=3, color="gray", linestyle="--", alpha=0.55)
    axes[0].set_title("Linear Probe: Truth Class (T/F/U) - Test ID", fontsize=11)
    axes[0].set_xlabel("Layer (0..6)", fontsize=10)
    axes[0].set_ylabel("Probe Accuracy", fontsize=10)
    axes[0].set_xticks(layers)
    axes[0].legend(fontsize=8.5, loc="upper left")
    axes[0].grid(alpha=0.3)

    for name, curve in rep_all["probe_qdep_ood"].items():
        axes[1].plot(layers, curve, marker="o", linewidth=2, label=pretty_name(name))
    axes[1].axvline(x=3, color="gray", linestyle="--", alpha=0.55)
    axes[1].set_title("Linear Probe: Proof Depth QDep - Test OOD", fontsize=11)
    axes[1].set_xlabel("Layer (0..6)", fontsize=10)
    axes[1].set_ylabel("Probe Accuracy", fontsize=10)
    axes[1].set_xticks(layers)
    axes[1].legend(fontsize=8.5, loc="upper left")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "probing_curves.png"), dpi=220)
    plt.close()

    # 3. CKA Diagonal Profiles (opzionale ma pulito)
    plt.figure(figsize=(7.5, 4.6))
    for pair_name, cka_diag in rep_all["cka_sentence_diag_id"].items():
        if (
            pair_name.endswith("_vs_free")
            or pair_name.startswith("free_vs_")
            or pair_name.startswith("base_pretrained_vs_const")
        ):
            plt.plot(layers, cka_diag, marker="s", linewidth=2, label=pair_name)
    plt.axvline(x=3, color="gray", linestyle="--", alpha=0.6)
    plt.title("Layer-wise Linear CKA (Sentence Representations - Test ID)", fontsize=11)
    plt.xlabel("Layer (0..6)", fontsize=10)
    plt.ylabel("Linear CKA", fontsize=10)
    plt.xticks(layers)
    plt.ylim(0.0, 1.05)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cka_profiles.png"), dpi=220)
    plt.close()

    # 4. Full 7x7 CKA Heatmaps
    cka_full = rep_cs["cka_matrices_full"]
    items = list(cka_full.items())
    n = len(items)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(10.2, 4.5 * rows), constrained_layout=True)
    axes_flat = np.array(axes).reshape(-1)

    for idx, (raw_title, mat_list) in enumerate(items):
        mat = np.array(mat_list)
        ax = axes_flat[idx]
        im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis", origin="lower")
        title = PRETTY_CKA_TITLES.get(raw_title, raw_title)
        ax.set_title(title, fontsize=10.5, pad=6)
        ax.set_xlabel("Model B Layer (0..6)", fontsize=9.5)
        ax.set_ylabel("Model A Layer (0..6)", fontsize=9.5)
        ax.set_xticks(range(7))
        ax.set_yticks(range(7))

        for i in range(7):
            for j in range(7):
                val = mat[i, j]
                color = "black" if val > 0.6 else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7.5)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.colorbar(im, ax=axes_flat[:n].tolist(), shrink=0.8, aspect=25)
    plt.savefig(os.path.join(out_dir, "cka_matrices_7x7.png"), dpi=220)
    plt.close()

    print(f"Tutti i grafici in inglese salvati in: {out_dir}")


if __name__ == "__main__":
    main()