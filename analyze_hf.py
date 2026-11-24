import argparse
import copy
import json
import os
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import LABEL_MAP
from train import load_or_build_datasets, set_seed
from train_hf import (
    TextAlphaPermutator,
    compute_sentence_ids,
    load_or_build_natlang,
    pool_sentences,
)


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Linear CKA (Kornblith et al., 2019) tra matrici (N, D) in FP64 per precisione.
    """
    X = X.double()
    Y = Y.double()
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    ytx = Y.t() @ X
    xtx = X.t() @ X
    yty = Y.t() @ Y

    num = (ytx * ytx).sum()
    den = torch.sqrt((xtx * xtx).sum() * (yty * yty).sum()).clamp(min=1e-12)
    return float((num / den).item())


def find_models_in_dir(save_dir: str, seed: int = 42) -> Dict[str, str]:
    found = {}
    if not os.path.exists(save_dir):
        return found
    for name in sorted(os.listdir(save_dir)):
        if "_debug" in name:
            continue
        path = os.path.join(save_dir, name, "best_model.pt")
        if not os.path.exists(path):
            continue
        if f"seed{seed}" not in name:
            continue
        if name.startswith("free_aug_"):
            found["free_aug"] = path
        elif name.startswith("free_"):
            found["free"] = path
        elif name.startswith("constrained_"):
            found["constrained"] = path
    return found


def load_hf_model(model_name: str, ckpt_path: str, device: torch.device):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3
    ).to(device)
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def extract_features_and_preds(
    model,
    records: List[dict],
    hf_tok,
    permutator: TextAlphaPermutator,
    dot_id: int,
    device: torch.device,
    batch_size: int = 64,
    compute_perm: bool = True,
    perm_seed: int = 777,
    max_sent_keep: int = 12000,
) -> Dict[str, object]:
    """Estrae predizioni, stati [CLS] per tutti i 7 layer (0..6) e stati delle frasi."""
    rng = random.Random(perm_seed)
    n_layers_plus_emb = 7  # emb + 6 layers per distilroberta-base

    preds_all = []
    cls_by_layer = [[] for _ in range(n_layers_plus_emb)]
    sent_x_by_layer = [[] for _ in range(n_layers_plus_emb)]
    sent_px_by_layer = [[] for _ in range(n_layers_plus_emb)]
    sent_count = 0

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        qs = [r["question"] for r in batch]
        ths = [r["theory"] for r in batch]

        enc = hf_tok(
            qs, ths, padding=True, truncation=True, max_length=384, return_tensors="pt"
        )
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
            out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        preds_all.append(out.logits.argmax(dim=-1).cpu())

        for l in range(len(out.hidden_states)):
            cls_by_layer[l].append(out.hidden_states[l][:, 0, :].float().cpu())

        if compute_perm and sent_count < max_sent_keep:
            p_pairs = [permutator.permute_pair(q, t, rng) for q, t in zip(qs, ths)]
            penc = hf_tok(
                [p[0] for p in p_pairs],
                [p[1] for p in p_pairs],
                padding=True,
                truncation=True,
                max_length=384,
                return_tensors="pt",
            )
            pids = penc["input_ids"].to(device)
            pmask = penc["attention_mask"].to(device)

            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                pout = model(
                    input_ids=pids, attention_mask=pmask, output_hidden_states=True
                )

            sx = compute_sentence_ids(
                ids, mask, dot_id, hf_tok.cls_token_id, hf_tok.sep_token_id
            )
            spx = compute_sentence_ids(
                pids, pmask, dot_id, hf_tok.cls_token_id, hf_tok.sep_token_id
            )

            for l in range(len(out.hidden_states)):
                rx, vx = pool_sentences(out.hidden_states[l], sx)
                rpx, vpx = pool_sentences(pout.hidden_states[l], spx)
                s = min(rx.size(1), rpx.size(1))
                m = vx[:, :s] & vpx[:, :s]
                sent_x_by_layer[l].append(rx[:, :s, :][m].cpu())
                sent_px_by_layer[l].append(rpx[:, :s, :][m].cpu())

            sent_count += int(m.sum().item())

    return {
        "preds": torch.cat(preds_all, dim=0),
        "cls": [torch.cat(chunks, dim=0) for chunks in cls_by_layer],
        "sent_x": (
            [torch.cat(chunks, dim=0) for chunks in sent_x_by_layer]
            if compute_perm
            else None
        ),
        "sent_px": (
            [torch.cat(chunks, dim=0) for chunks in sent_px_by_layer]
            if compute_perm
            else None
        ),
    }


def compute_invariance_curve(feat: Dict[str, object]) -> List[float]:
    """Centered Cosine Similarity tra x e perm(x) per ciascuno strato 0..6."""
    curve = []
    for l in range(len(feat["sent_x"])):
        vx = feat["sent_x"][l]
        vpx = feat["sent_px"][l]
        mu = torch.cat([vx, vpx], dim=0).mean(dim=0, keepdim=True)
        cos = F.cosine_similarity(vx - mu, vpx - mu, dim=-1).mean().item()
        curve.append(float(cos))
    return curve


def run_linear_probe(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    num_classes: int,
    device: torch.device,
    epochs: int = 80,
) -> float:
    """Linear probe veloce su GPU per misurare cosa è leggibile linearmente a uno strato."""
    D = X_train.size(1)
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True).clamp(min=1e-5)

    Xtr = ((X_train - mean) / std).to(device)
    Xte = ((X_test - mean) / std).to(device)
    ytr = y_train.to(device)
    yte = y_test.to(device)

    probe = nn.Linear(D, num_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=1e-3)

    probe.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(probe(Xtr), ytr)
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        acc = (probe(Xte).argmax(dim=-1) == yte).float().mean().item()
    return float(acc)


def run_model_stitching(
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
    """
    Innesto: primi `split_layer` strati di A -> (adattatore lineare opzionale) -> ultimi strati di B.
    Usa forward hook su encoder.layer[split_layer - 1] per compatibilità totale con transformers.
    """
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
                [LABEL_MAP[r["answer"]] for r in batch],
                dtype=torch.long,
                device=device,
            )

            with torch.no_grad():
                out_a = model_a(
                    input_ids=ids, attention_mask=mask, output_hidden_states=True
                )
                h_a = out_a.hidden_states[split_layer]
                h_adapted = adapter(h_a.float()).to(h_a.dtype)

                def hook_fn(module, inputs, output):
                    if isinstance(output, tuple):
                        return (h_adapted,) + output[1:]
                    return h_adapted

                handle = model_b.roberta.encoder.layer[
                    split_layer - 1
                ].register_forward_hook(hook_fn)
                try:
                    out_b = model_b(input_ids=ids, attention_mask=mask)
                finally:
                    handle.remove()

                preds = out_b.logits.argmax(dim=-1)
                correct += int((preds == labels).sum().item())
                total += labels.size(0)
        return correct / max(total, 1)

    # 1. Zero-shot stitching (Identity)
    zero_shot_acc = eval_stitched(nn.Identity().to(device), test_records)

    # 2. Linear-adapted stitching (alleniamo solo adapter 768->768 su train_records)
    adapter = nn.Linear(D, D).to(device)
    nn.init.eye_(adapter.weight)
    nn.init.zeros_(adapter.bias)
    opt = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)

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
                [LABEL_MAP[r["answer"]] for r in batch],
                dtype=torch.long,
                device=device,
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

            handle = model_b.roberta.encoder.layer[
                split_layer - 1
            ].register_forward_hook(hook_fn)
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


def make_plots(report: dict, out_dir: str):
    # 1. Curva Invarianza Simbolica (Layer 0..6)
    plt.figure(figsize=(8, 5))
    layers = list(range(7))
    for name, curve in report["invariance_curves_id"].items():
        plt.plot(layers, curve, marker="o", linewidth=2, label=name)
    plt.axvline(x=3, color="gray", linestyle="--", alpha=0.6, label="Layer vincolato (L3)")
    plt.title("Invarianza Simbolica per Strato (Test ID - Centered Cosine)")
    plt.xlabel("Strato (0=Embeddings, 1..6=Transformer Layers)")
    plt.ylabel("Centered Cosine Similarity (x vs perm(x))")
    plt.ylim(0.5, 1.02)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "invariance_curve.png"), dpi=180)
    plt.close()

    # 2. CKA Diagonal Profile rispetto a 'free' e 'base_pretrained'
    plt.figure(figsize=(8, 5))
    for pair_name, cka_diag in report["cka_sentence_diag_id"].items():
        plt.plot(layers, cka_diag, marker="s", linewidth=2, label=pair_name)
    plt.axvline(x=3, color="gray", linestyle="--", alpha=0.6)
    plt.title("Linear CKA Strato-per-Strato (Rappresentazioni Frasi - Test ID)")
    plt.xlabel("Strato (0..6)")
    plt.ylabel("Linear CKA")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cka_profiles.png"), dpi=180)
    plt.close()

    # 3. Linear Probing per Classe e QDep
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, curve in report["probe_cls_id"].items():
        axes[0].plot(layers, curve, marker="o", linewidth=2, label=name)
    axes[0].set_title("Linear Probe Accuracy: Classe (T/F/U) - Test ID")
    axes[0].set_xlabel("Strato (0..6)")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for name, curve in report["probe_qdep_ood"].items():
        axes[1].plot(layers, curve, marker="o", linewidth=2, label=name)
    axes[1].set_title("Linear Probe Accuracy: Profondita Logica QDep (OOD)")
    axes[1].set_xlabel("Strato (0..6)")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "probing_curves.png"), dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="distilroberta-base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--cache_dir", type=str, default="data_cache")
    parser.add_argument("--n_samples", type=int, default=1500)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_map = find_models_in_dir(args.save_dir, seed=args.seed)
    print("Modelli trovati:", ckpt_map)
    assert len(ckpt_map) >= 2, "Servono almeno 2 modelli allenati in --save_dir!"

    analysis_dir = os.path.join(args.save_dir, f"analysis_seed{args.seed}")
    os.makedirs(analysis_dir, exist_ok=True)

    _, train_ds, val_ds, test_id_ds, test_ood_ds = load_or_build_datasets(
        args.cache_dir, data_seed=args.data_seed
    )
    train_theories = set(r["theory"] for r in train_ds.records)
    natlang_records = load_or_build_natlang(
        args.cache_dir, train_theories, max_samples=3000, seed=args.data_seed
    )

    N = args.n_samples
    id_eval = test_id_ds.records[:N]
    ood_eval = test_ood_ds.records[:N]
    nat_eval = natlang_records[:N]

    hf_tok = AutoTokenizer.from_pretrained(args.model_name)
    dot_id = hf_tok.encode(".", add_special_tokens=False)[0]
    permutator = TextAlphaPermutator()

    # Includiamo anche il modello base non allenato come riferimento!
    model_specs = {"base_pretrained": None}
    model_specs.update(ckpt_map)

    feats_id = {}
    feats_ood = {}
    preds_nat = {}
    loaded_models = {}

    for name, ckpt_path in model_specs.items():
        print(f"Estrazione feature per '{name}'...")
        m = load_hf_model(args.model_name, ckpt_path, device)
        loaded_models[name] = m
        feats_id[name] = extract_features_and_preds(
            m, id_eval, hf_tok, permutator, dot_id, device, compute_perm=True
        )
        feats_ood[name] = extract_features_and_preds(
            m, ood_eval, hf_tok, permutator, dot_id, device, compute_perm=True
        )
        if name != "base_pretrained":
            preds_nat[name] = extract_features_and_preds(
                m, nat_eval, hf_tok, permutator, dot_id, device, compute_perm=False
            )["preds"]

    # 1. Curve di Invarianza Simbolica (layer 0..6)
    inv_curves_id = {
        name: compute_invariance_curve(feats_id[name]) for name in model_specs
    }
    inv_curves_ood = {
        name: compute_invariance_curve(feats_ood[name]) for name in model_specs
    }

    # 2. CKA strato-per-strato (Sentence & CLS)
    trained_names = [k for k in model_specs if k != "base_pretrained"]
    pairs = []
    for idx_a in range(len(trained_names)):
        pairs.append(("base_pretrained", trained_names[idx_a]))
        for idx_b in range(idx_a + 1, len(trained_names)):
            pairs.append((trained_names[idx_a], trained_names[idx_b]))

    cka_sent_id = {}
    cka_cls_id = {}
    cka_sent_ood = {}
    for a, b in pairs:
        key = f"{a}_vs_{b}"
        cka_sent_id[key] = [
            linear_cka(feats_id[a]["sent_x"][l], feats_id[b]["sent_x"][l])
            for l in range(7)
        ]
        cka_cls_id[key] = [
            linear_cka(feats_id[a]["cls"][l], feats_id[b]["cls"][l])
            for l in range(7)
        ]
        cka_sent_ood[key] = [
            linear_cka(feats_ood[a]["sent_x"][l], feats_ood[b]["sent_x"][l])
            for l in range(7)
        ]

    # 3. Disaccordo a coppie sulle predizioni
    disagreements = {}
    y_id = torch.tensor([LABEL_MAP[r["answer"]] for r in id_eval])
    y_ood = torch.tensor([LABEL_MAP[r["answer"]] for r in ood_eval])
    y_nat = torch.tensor([LABEL_MAP[r["answer"]] for r in nat_eval])

    for i in range(len(trained_names)):
        for j in range(i + 1, len(trained_names)):
            a, b = trained_names[i], trained_names[j]
            pair_res = {}
            for split_tag, p_a, p_b, y_true in [
                ("id", feats_id[a]["preds"], feats_id[b]["preds"], y_id),
                ("ood_depth", feats_ood[a]["preds"], feats_ood[b]["preds"], y_ood),
                ("natlang", preds_nat[a], preds_nat[b], y_nat),
            ]:
                dis = float((p_a != p_b).float().mean().item())
                a_only = int(((p_a == y_true) & (p_b != y_true)).sum().item())
                b_only = int(((p_b == y_true) & (p_a != y_true)).sum().item())
                pair_res[split_tag] = {
                    "disagreement_rate": dis,
                    f"{a}_only_right": a_only,
                    f"{b}_only_right": b_only,
                }
            disagreements[f"{a}_vs_{b}"] = pair_res

    # 4. Linear Probing per Strato (Classe su ID, QDep su OOD True/False)
    print("Esecuzione Linear Probing strato per strato...")
    probe_cls_id = {}
    probe_qdep_ood = {}
    half = N // 2
    tf_mask_ood = y_ood != 2
    qdep_ood = torch.tensor([min(int(r["QDep"]), 5) for r in ood_eval])
    tf_idx = tf_mask_ood.nonzero(as_tuple=True)[0]
    tf_half = len(tf_idx) // 2

    for name in model_specs:
        cls_curve = []
        qdep_curve = []
        for l in range(7):
            acc_c = run_linear_probe(
                feats_id[name]["cls"][l][:half],
                y_id[:half],
                feats_id[name]["cls"][l][half:],
                y_id[half:],
                num_classes=3,
                device=device,
            )
            cls_curve.append(acc_c)

            tf_feats = feats_ood[name]["cls"][l][tf_idx]
            tf_q = qdep_ood[tf_idx]
            acc_q = run_linear_probe(
                tf_feats[:tf_half],
                tf_q[:tf_half],
                tf_feats[tf_half:],
                tf_q[tf_half:],
                num_classes=6,
                device=device,
            )
            qdep_curve.append(acc_q)
        probe_cls_id[name] = cls_curve
        probe_qdep_ood[name] = qdep_curve

    # 5. Model Stitching (Layer 3 -> Layer 4..6)
    report = {
        "models": list(model_specs.keys()),
        "invariance_curves_id": inv_curves_id,
        "invariance_curves_ood": inv_curves_ood,
        "cka_sentence_diag_id": cka_sent_id,
        "cka_cls_diag_id": cka_cls_id,
        "cka_sentence_diag_ood": cka_sent_ood,
        "disagreements": disagreements,
        "probe_cls_id": probe_cls_id,
        "probe_qdep_ood": probe_qdep_ood,
        "stitching_layer3": {},
    }

    with open(os.path.join(analysis_dir, "analysis_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    make_plots(report, analysis_dir)
    print(f"Report parziale e grafici PNG salvati in: {analysis_dir}")

    # 5. Model Stitching (Layer 3 -> Layer 4..6)
    print("Esecuzione Model Stitching (layer 3)...")
    stitching = {}
    stitch_train = train_ds.records[:1000]
    stitch_test = test_id_ds.records[:600]
    for a in trained_names:
        for b in trained_names:
            if a == b:
                continue
            print(f"  Stitching {a}[:3] -> {b}[3:]...")
            stitching[f"{a}_to_{b}"] = run_model_stitching(
                loaded_models[a],
                loaded_models[b],
                stitch_train,
                stitch_test,
                hf_tok,
                device,
                split_layer=3,
                adapt_epochs=2,
            )

    report["stitching_layer3"] = stitching
    with open(os.path.join(analysis_dir, "analysis_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== RIEPILOGO ANALISI ===")
    print("Invarianza Simbolica ID (L0..L6):")
    for k, v in inv_curves_id.items():
        print(f"  {k:16s}: {[round(x, 3) for x in v]}")
    print("\nCKA Frasi ID (L0..L6):")
    for k, v in cka_sent_id.items():
        print(f"  {k:28s}: {[round(x, 3) for x in v]}")
    print("\nModel Stitching (Layer 3):")
    for k, v in stitching.items():
        print(f"  {k:24s}: {v}")
    print(f"\nReport completo salvato in: {analysis_dir}")


if __name__ == "__main__":
    main()