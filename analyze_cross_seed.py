import argparse
import json
import os
import random
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from analyze_hf import (
    compute_invariance_curve,
    extract_features_and_preds,
    linear_cka,
    load_hf_model,
)
from data import LABEL_MAP
from train import load_or_build_datasets, set_seed
from train_hf import TextAlphaPermutator


def compute_cka_matrix(feats_a: List[torch.Tensor], feats_b: List[torch.Tensor]) -> np.ndarray:
    """Calcola la matrice CKA 7x7 completa tra tutti gli strati di A e tutti gli strati di B."""
    L = len(feats_a)
    mat = np.zeros((L, L), dtype=np.float64)
    for i in range(L):
        for j in range(L):
            mat[i, j] = linear_cka(feats_a[i], feats_b[j])
    return mat


def run_sklearn_probe(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    C: float = 0.1,
) -> float:
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True).clamp(min=1e-5)

    Xtr = ((X_train - mean) / std).numpy()
    Xte = ((X_test - mean) / std).numpy()
    clf = LogisticRegression(C=C, max_iter=500)
    clf.fit(Xtr, y_train.numpy())
    preds = clf.predict(Xte)
    return float((preds == y_test.numpy()).mean())


def run_stitching_reg(
    model_a,
    model_b,
    train_records: List[dict],
    test_records: List[dict],
    hf_tok,
    device: torch.device,
    split_layer: int = 3,
    adapt_epochs: int = 3,
    batch_size: int = 32,
) -> Dict[str, float]:
    model_a.eval()
    model_b.eval()
    D = model_a.config.hidden_size

    def eval_stitched(adapter: nn.Module, recs: List[dict]) -> float:
        adapter.eval()
        correct = 0
        total = 0
        for i in range(0, len(recs), batch_size):
            batch = recs[i : i + batch_size]
            enc = hf_tok(
                [r["question"] for r in batch],
                [r["theory"] for r in batch],
                padding=True,
                truncation=True,
                max_length=384,
                return_tensors="pt",
            )
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            labels = torch.tensor(
                [LABEL_MAP[r["answer"]] for r in batch], dtype=torch.long, device=device
            )

            with torch.no_grad():
                h_a = model_a(
                    input_ids=ids, attention_mask=mask, output_hidden_states=True
                ).hidden_states[split_layer]
                h_adapted = adapter(h_a.float()).to(h_a.dtype)

                def hook_fn(module, inputs, output):
                    if isinstance(output, tuple):
                        return (h_adapted,) + output[1:]
                    return h_adapted

                handle = model_b.roberta.encoder.layer[split_layer - 1].register_forward_hook(hook_fn)
                try:
                    out_b = model_b(input_ids=ids, attention_mask=mask)
                finally:
                    handle.remove()

                correct += int((out_b.logits.argmax(dim=-1) == labels).sum().item())
                total += labels.size(0)
        return correct / max(total, 1)

    zero_shot_acc = eval_stitched(nn.Identity().to(device), test_records)

    adapter = nn.Linear(D, D).to(device)
    nn.init.eye_(adapter.weight)
    nn.init.zeros_(adapter.bias)
    opt = torch.optim.AdamW(adapter.parameters(), lr=5e-4, weight_decay=1e-2)

    for _ in range(adapt_epochs):
        adapter.train()
        shuffled = train_records.copy()
        random.shuffle(shuffled)
        for i in range(0, len(shuffled), batch_size):
            batch = shuffled[i : i + batch_size]
            enc = hf_tok(
                [r["question"] for r in batch],
                [r["theory"] for r in batch],
                padding=True,
                truncation=True,
                max_length=384,
                return_tensors="pt",
            )
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            labels = torch.tensor(
                [LABEL_MAP[r["answer"]] for r in batch], dtype=torch.long, device=device
            )

            with torch.no_grad():
                h_a = model_a(
                    input_ids=ids, attention_mask=mask, output_hidden_states=True
                ).hidden_states[split_layer]

            h_adapted = adapter(h_a.float())

            def hook_fn(module, inputs, output):
                if isinstance(output, tuple):
                    return (h_adapted,) + output[1:]
                return h_adapted

            handle = model_b.roberta.encoder.layer[split_layer - 1].register_forward_hook(hook_fn)
            try:
                out_b = model_b(input_ids=ids, attention_mask=mask)
            finally:
                handle.remove()

            loss = F.cross_entropy(out_b.logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

    adapted_acc = eval_stitched(adapter, test_records)
    return {
        "zero_shot_acc": float(zero_shot_acc),
        "linear_adapted_acc": float(adapted_acc),
    }


def plot_cka_matrices(matrices: Dict[str, np.ndarray], out_path: str):
    items = list(matrices.items())
    n = len(items)
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(11, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for idx, (title, mat) in enumerate(items):
        ax = axes[idx]
        im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis", origin="lower")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Strato Modello B (0..6)")
        ax.set_ylabel("Strato Modello A (0..6)")
        ax.set_xticks(range(7))
        ax.set_yticks(range(7))

        for i in range(7):
            for j in range(7):
                val = mat[i, j]
                color = "black" if val > 0.6 else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7.5)

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    fig.colorbar(im, ax=axes[:n].tolist(), fraction=0.025, pad=0.03)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="distilroberta-base")
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--cache_dir", type=str, default="data_cache")
    parser.add_argument("--n_samples", type=int, default=1500)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidates = {
        "free_s42": os.path.join(args.save_dir, "free_seed42", "best_model.pt"),
        "free_s123": os.path.join(args.save_dir, "free_seed123", "best_model.pt"),
        "const05_s42": os.path.join(args.save_dir, "constrained_seed42_l3_lam0.5", "best_model.pt"),
        "const05_s123": os.path.join(args.save_dir, "constrained_seed123_l3_lam0.5", "best_model.pt"),
        "const10_s42": os.path.join(args.save_dir, "constrained_seed42_l3_lam1.0", "best_model.pt"),
    }
    models_found = {k: v for k, v in candidates.items() if os.path.exists(v)}
    print("Modelli trovati:", models_found)

    out_dir = os.path.join(args.save_dir, "analysis_cross_seed")
    os.makedirs(out_dir, exist_ok=True)

    _, train_ds, _, test_id_ds, test_ood_ds = load_or_build_datasets(
        args.cache_dir, data_seed=args.data_seed
    )
    N = args.n_samples
    id_eval = test_id_ds.records[:N]
    ood_eval = test_ood_ds.records[:N]

    hf_tok = AutoTokenizer.from_pretrained(args.model_name)
    dot_id = hf_tok.encode(".", add_special_tokens=False)[0]
    permutator = TextAlphaPermutator()

    feats_id = {}
    feats_ood = {}
    loaded = {}
    for name, path in models_found.items():
        print(f"Estrazione feature '{name}'...")
        m = load_hf_model(args.model_name, path, device)
        loaded[name] = m
        feats_id[name] = extract_features_and_preds(
            m, id_eval, hf_tok, permutator, dot_id, device, compute_perm=True
        )
        feats_ood[name] = extract_features_and_preds(
            m, ood_eval, hf_tok, permutator, dot_id, device, compute_perm=False
        )

    # 1. Matrici CKA 7x7 complete
    pair_specs = [
        ("free_s42 vs free_s123 (Controllo Seed)", "free_s42", "free_s123"),
        ("const05_s42 vs const05_s123 (Controllo Seed)", "const05_s42", "const05_s123"),
        ("free_s42 vs const05_s42 (Libera vs Vincolata 0.5)", "free_s42", "const05_s42"),
        ("free_s42 vs const10_s42 (Libera vs Vincolata 1.0)", "free_s42", "const10_s42"),
    ]
    cka_mats = {}
    for title, a, b in pair_specs:
        if a in feats_id and b in feats_id:
            cka_mats[title] = compute_cka_matrix(feats_id[a]["sent_x"], feats_id[b]["sent_x"])

    plot_cka_matrices(cka_mats, os.path.join(out_dir, "cka_matrices_7x7.png"))

    # 2. Invarianza + Probing regolarizzato (LogisticRegression)
    y_id = torch.tensor([LABEL_MAP[r["answer"]] for r in id_eval])
    y_ood = torch.tensor([LABEL_MAP[r["answer"]] for r in ood_eval])
    half = N // 2
    tf_idx = (y_ood != 2).nonzero(as_tuple=True)[0]
    qdep_ood = torch.tensor([min(int(r["QDep"]), 5) for r in ood_eval])
    tf_half = len(tf_idx) // 2

    inv_curves = {}
    probe_cls = {}
    probe_qdep = {}
    for name in models_found:
        inv_curves[name] = compute_invariance_curve(feats_id[name])
        probe_cls[name] = [
            run_sklearn_probe(
                feats_id[name]["cls"][l][:half],
                y_id[:half],
                feats_id[name]["cls"][l][half:],
                y_id[half:],
            )
            for l in range(7)
        ]
        probe_qdep[name] = [
            run_sklearn_probe(
                feats_ood[name]["cls"][l][tf_idx][:tf_half],
                qdep_ood[tf_idx][:tf_half],
                feats_ood[name]["cls"][l][tf_idx][tf_half:],
                qdep_ood[tf_idx][tf_half:],
            )
            for l in range(7)
        ]

    # 3. Stitching pulito (3000 esempi, regolarizzato)
    stitch_pairs = [
        ("free_s42", "free_s123"),
        ("free_s123", "free_s42"),
        ("free_s42", "const05_s42"),
        ("const05_s42", "free_s42"),
        ("free_s123", "const05_s123"),
        ("const05_s123", "free_s123"),
    ]
    stitching = {}
    st_train = train_ds.records[:3000]
    st_test = test_id_ds.records[:800]
    for a, b in stitch_pairs:
        if a in loaded and b in loaded:
            print(f"Stitching {a} -> {b}...")
            stitching[f"{a}_to_{b}"] = run_stitching_reg(
                loaded[a], loaded[b], st_train, st_test, hf_tok, device
            )

    report = {
        "models_found": list(models_found.keys()),
        "cka_matrices_diag": {k: np.diag(v).tolist() for k, v in cka_mats.items()},
        "cka_matrices_full": {k: v.tolist() for k, v in cka_mats.items()},
        "invariance_curves": inv_curves,
        "probe_cls_sklearn": probe_cls,
        "probe_qdep_sklearn": probe_qdep,
        "stitching_reg": stitching,
    }
    with open(os.path.join(out_dir, "cross_seed_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== RIEPILOGO CROSS-SEED ===")
    for k, diag in report["cka_matrices_diag"].items():
        print(f"CKA diag | {k}: {[round(x, 3) for x in diag]}")
    print("\nStitching (regolarizzato):")
    for k, v in stitching.items():
        print(f"  {k}: {v}")
    print(f"\nSalvato tutto in: {out_dir}")


if __name__ == "__main__":
    main()