import argparse
import json
import os
import random
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from analyze_hf import linear_cka, load_hf_model
from data import LABEL_MAP
from train import set_seed, load_or_build_datasets
from train_hf import TextAlphaPermutator


class ResidualLinearAdapter(nn.Module):
    """Adapter h + delta(h) inizializzato all'identità esatta, così weight_decay tira verso I."""

    def __init__(self, d_model: int):
        super().__init__()
        self.delta = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.delta(x)


@torch.no_grad()
def compute_token_level_cka(
    models: Dict[str, nn.Module],
    records: List[dict],
    tokenizer,
    device: torch.device,
    layer_idx: int = 3,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Calcola Token-level CKA al Layer 3 su token validi non-PAD (campionati fino a 15k token)."""
    feats = {k: [] for k in models}
    rng = random.Random(42)

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        enc = tokenizer(
            [r["question"] for r in batch],
            [r["theory"] for r in batch],
            padding=True,
            truncation=True,
            max_length=384,
            return_tensors="pt",
        )
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        valid_idx = (mask == 1).nonzero(as_tuple=False)
        # Campioniamo 1500 token casuali per batch per tenere la matrice compatta
        perm = torch.randperm(valid_idx.size(0))[:1500]
        sel = valid_idx[perm]
        b_idx, t_idx = sel[:, 0], sel[:, 1]

        for name, m in models.items():
            m.eval()
            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                h = m(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[layer_idx]
            feats[name].append(h[b_idx, t_idx].float().cpu())

    cat_feats = {k: torch.cat(v, dim=0) for k, v in feats.items()}
    res = {}
    pairs = [
        ("free_s42", "free_s123"),
        ("free_s42", "const05_s42"),
        ("free_s42", "const10_s42"),
    ]
    for a, b in pairs:
        if a in cat_feats and b in cat_feats:
            res[f"{a}_vs_{b}"] = linear_cka(cat_feats[a], cat_feats[b])
    return res


@torch.no_grad()
def eval_stitched_zeroshot_ood(
    model_a,
    model_b,
    ood_records: List[dict],
    unk_ood_records: List[dict],
    tokenizer,
    permutator: TextAlphaPermutator,
    device: torch.device,
    split_layer: int = 3,
    batch_size: int = 64,
) -> Dict[str, float]:
    model_a.eval()
    model_b.eval()
    rng = random.Random(202)

    def run_forward(recs, do_flip=False):
        preds_all = []
        p_preds_all = []
        for i in range(0, len(recs), batch_size):
            batch = recs[i : i + batch_size]
            qs = [r["question"] for r in batch]
            ths = [r["theory"] for r in batch]
            enc = tokenizer(qs, ths, padding=True, truncation=True, max_length=384, return_tensors="pt")
            ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)

            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                if model_a is model_b:
                    logits = model_a(input_ids=ids, attention_mask=mask).logits
                else:
                    h_a = model_a(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[split_layer]
                    def hook_fn(module, inputs, output):
                        return (h_a,) + output[1:] if isinstance(output, tuple) else h_a
                    handle = model_b.roberta.encoder.layer[split_layer - 1].register_forward_hook(hook_fn)
                    try:
                        logits = model_b(input_ids=ids, attention_mask=mask).logits
                    finally:
                        handle.remove()
            preds_all.append(logits.argmax(dim=-1).cpu())

            if do_flip:
                p_pairs = [permutator.permute_pair(q, t, rng) for q, t in zip(qs, ths)]
                penc = tokenizer([p[0] for p in p_pairs], [p[1] for p in p_pairs], padding=True, truncation=True, max_length=384, return_tensors="pt")
                pids, pmask = penc["input_ids"].to(device), penc["attention_mask"].to(device)
                with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                    if model_a is model_b:
                        plogits = model_a(input_ids=pids, attention_mask=pmask).logits
                    else:
                        h_pa = model_a(input_ids=pids, attention_mask=pmask, output_hidden_states=True).hidden_states[split_layer]
                        def phook_fn(module, inputs, output):
                            return (h_pa,) + output[1:] if isinstance(output, tuple) else h_pa
                        handle = model_b.roberta.encoder.layer[split_layer - 1].register_forward_hook(phook_fn)
                        try:
                            plogits = model_b(input_ids=pids, attention_mask=pmask).logits
                        finally:
                            handle.remove()
                p_preds_all.append(plogits.argmax(dim=-1).cpu())

        preds = torch.cat(preds_all, dim=0)
        p_preds = torch.cat(p_preds_all, dim=0) if do_flip else None
        return preds, p_preds

    # 1. OOD Acc + Flip Rate
    y_ood = torch.tensor([LABEL_MAP[r["answer"]] for r in ood_records])
    preds_ood, p_preds_ood = run_forward(ood_records, do_flip=True)
    ood_acc = float((preds_ood == y_ood).float().mean().item())
    ood_flip = float((preds_ood != p_preds_ood).float().mean().item())

    # 2. OOD Unknown FPR
    preds_unk, _ = run_forward(unk_ood_records, do_flip=False)
    unk_acc = float((preds_unk == 2).float().mean().item())

    return {
        "ood_acc": ood_acc,
        "ood_flip_rate": ood_flip,
        "ood_unknown_acc": unk_acc,
        "ood_unknown_fpr": 1.0 - unk_acc,
    }


def run_residual_stitching_large(
    model_a,
    model_b,
    train_records: List[dict],
    test_records: List[dict],
    tokenizer,
    device: torch.device,
    split_layer: int = 3,
    epochs: int = 2,
    batch_size: int = 64,
) -> Dict[str, float]:
    model_a.eval()
    model_b.eval()
    D = model_a.config.hidden_size
    adapter = ResidualLinearAdapter(D).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=3e-4, weight_decay=0.05)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    def eval_acc(adp):
        adp.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i in range(0, len(test_records), 64):
                batch = test_records[i : i + 64]
                enc = tokenizer([r["question"] for r in batch], [r["theory"] for r in batch], padding=True, truncation=True, max_length=384, return_tensors="pt")
                ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
                labels = torch.tensor([LABEL_MAP[r["answer"]] for r in batch], dtype=torch.long, device=device)
                with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                    h_a = model_a(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[split_layer]
                    h_adp = adp(h_a)
                    def hook_fn(module, inputs, output):
                        return (h_adp,) + output[1:] if isinstance(output, tuple) else h_adp
                    handle = model_b.roberta.encoder.layer[split_layer - 1].register_forward_hook(hook_fn)
                    try:
                        logits = model_b(input_ids=ids, attention_mask=mask).logits
                    finally:
                        handle.remove()
                correct += int((logits.argmax(dim=-1) == labels).sum().item())
                total += labels.size(0)
        return correct / total

    zs_acc = eval_acc(nn.Identity().to(device))

    for _ in range(epochs):
        adapter.train()
        shuffled = train_records.copy()
        random.shuffle(shuffled)
        for i in range(0, len(shuffled), batch_size):
            batch = shuffled[i : i + batch_size]
            enc = tokenizer([r["question"] for r in batch], [r["theory"] for r in batch], padding=True, truncation=True, max_length=384, return_tensors="pt")
            ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
            labels = torch.tensor([LABEL_MAP[r["answer"]] for r in batch], dtype=torch.long, device=device)

            with torch.no_grad():
                with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                    h_a = model_a(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[split_layer]

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                h_adp = adapter(h_a)
                def hook_fn(module, inputs, output):
                    return (h_adp,) + output[1:] if isinstance(output, tuple) else h_adp
                handle = model_b.roberta.encoder.layer[split_layer - 1].register_forward_hook(hook_fn)
                try:
                    logits = model_b(input_ids=ids, attention_mask=mask).logits
                finally:
                    handle.remove()
                loss = F.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

    adp_acc = eval_acc(adapter)
    return {"zero_shot_acc": float(zs_acc), "residual_adapted_acc": float(adp_acc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="distilroberta-base")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    permutator = TextAlphaPermutator()

    _, train_ds, _, test_id_ds, test_ood_ds = load_or_build_datasets("data_cache", data_seed=42)

    paths = {
        "free_s42": os.path.join(args.save_dir, "free_seed42", "best_model.pt"),
        "free_s123": os.path.join(args.save_dir, "free_seed123", "best_model.pt"),
        "const05_s42": os.path.join(args.save_dir, "constrained_seed42_l3_lam0.5", "best_model.pt"),
        "const05_s123": os.path.join(args.save_dir, "constrained_seed123_l3_lam0.5", "best_model.pt"),
        "const10_s42": os.path.join(args.save_dir, "constrained_seed42_l3_lam1.0", "best_model.pt"),
    }
    models = {k: load_hf_model(args.model_name, p, device) for k, p in paths.items() if os.path.exists(p)}

    print("=== 1. TOKEN-LEVEL CKA (LAYER 3, Test ID N=800) ===")
    tok_cka = compute_token_level_cka(models, test_id_ds.records[:800], tokenizer, device)
    print(json.dumps(tok_cka, indent=2))

    print("\n=== 2. CAUSAL ZERO-SHOT STITCHING SU OOD FLIP RATE & UNKNOWN FPR ===")
    ood_sub = test_ood_ds.records[:2500]
    unk_ood = [r for r in test_ood_ds.records if r["answer"] == "Unknown"]
    causal_specs = [
        ("free_s42 (baseline)", "free_s42", "free_s42"),
        ("const05_s42 (baseline)", "const05_s42", "const05_s42"),
        ("const10_s42 (baseline)", "const10_s42", "const10_s42"),
        ("const05_s42 -> free_s42", "const05_s42", "free_s42"),
        ("const05_s123 -> free_s123", "const05_s123", "free_s123"),
        ("const10_s42 -> free_s42", "const10_s42", "free_s42"),
        ("free_s42 -> const10_s42", "free_s42", "const10_s42"),
    ]
    causal_res = {}
    for label, a, b in causal_specs:
        if a in models and b in models:
            r = eval_stitched_zeroshot_ood(models[a], models[b], ood_sub, unk_ood, tokenizer, permutator, device)
            causal_res[label] = r
            print(f"  {label:28s} | OOD Acc={r['ood_acc']:.4f} | Flip={r['ood_flip_rate']:.4f} | Unk_FPR={r['ood_unknown_fpr']:.4f}")

    print("\n=== 3. RESIDUAL STITCHING SU N=20,000 ===")
    st_train = train_ds.records[:20000]
    st_test = test_id_ds.records[:1500]
    stitch_pairs = [
        ("free_s42", "free_s123"),
        ("free_s123", "free_s42"),
        ("free_s42", "const05_s42"),
        ("const05_s42", "free_s42"),
        ("free_s123", "const05_s123"),
        ("const05_s123", "free_s123"),
    ]
    res_stitch = {}
    for a, b in stitch_pairs:
        if a in models and b in models:
            r = run_residual_stitching_large(models[a], models[b], st_train, st_test, tokenizer, device)
            res_stitch[f"{a}_to_{b}"] = r
            print(f"  {a}_to_{b:15s} -> {r}")

    out_path = os.path.join(args.save_dir, "analysis_cross_seed", "causal_stitch_report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"token_cka": tok_cka, "causal_ood_stitch": causal_res, "residual_stitch_20k": res_stitch}, f, indent=2)
    print(f"\nSalvato in {out_path}")


if __name__ == "__main__":
    main()