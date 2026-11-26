import argparse
import json
import os
import re
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from analyze_hf import load_hf_model
from data import ANIMALS, PEOPLE, LABEL_MAP
from train import load_or_build_datasets, set_seed


def check_delimiters(records_list: List[List[dict]]):
    q_ends = set()
    has_qmark = 0
    has_excl = 0
    total = 0
    for recs in records_list:
        for r in recs:
            total += 1
            q = r["question"].strip()
            t = r["theory"].strip()
            if q:
                q_ends.add(q[-1])
            if "?" in q or "?" in t:
                has_qmark += 1
            if "!" in q or "!" in t:
                has_excl += 1
    print(f"=== CHECK DELIMITATORI ({total} esempi) ===")
    print(f"  Carattere finale delle query: {sorted(q_ends)}")
    print(f"  Occorrenze '?': {has_qmark} | Occorrenze '!': {has_excl}")


def print_unknown_from_eval_json(save_dir: str, folders: Dict[str, str]):
    print("\n=== DATI ESATTI SU 'UNKNOWN' DA eval_results.json (TUTTI I DATASET) ===")
    for label, folder in folders.items():
        p = os.path.join(save_dir, folder, "eval_results.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            res = json.load(f)["best_model"]
        id_u = res["test_id"]["acc_by_class"]["Unknown"]
        ood_u = res["test_ood_depth"]["acc_by_class"]["Unknown"]
        nat_u = res["test_ood_natlang"]["acc_by_class"]["Unknown"]
        print(
            f"  {label:18s} | ID Unknown={id_u:.4f} (FPR={1-id_u:.4f}) | "
            f"OOD Unknown={ood_u:.4f} (FPR={1-ood_u:.4f}) | NatLang Unknown={nat_u:.4f}"
        )


@torch.no_grad()
def eval_unknown_fp_breakdown(model, unk_records: List[dict], tokenizer, device, batch_size=64):
    model.eval()
    pred_f, pred_t, pred_u = 0, 0, 0
    for i in range(0, len(unk_records), batch_size):
        batch = unk_records[i : i + batch_size]
        enc = tokenizer(
            [r["question"] for r in batch],
            [r["theory"] for r in batch],
            padding=True,
            truncation=True,
            max_length=384,
            return_tensors="pt",
        )
        with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits
        preds = logits.argmax(dim=-1).cpu()
        pred_f += int((preds == 0).sum().item())
        pred_t += int((preds == 1).sum().item())
        pred_u += int((preds == 2).sum().item())
    n = len(unk_records)
    return {
        "n": n,
        "acc_u": pred_u / n,
        "fpr_total": (pred_f + pred_t) / n,
        "fp_to_false": pred_f / n,
        "fp_to_true": pred_t / n,
    }


@torch.no_grad()
def compute_raw_and_centered_ices(
    model,
    records: List[dict],
    tokenizer,
    device: torch.device,
    entity_vocab: List[str],
    target_layers: List[int] = [3, 6],
    max_eval: int = 400,
) -> Dict[int, Dict[str, float]]:
    model.eval()
    ent_regex = re.compile(
        r"\b(" + "|".join(re.escape(e) for e in entity_vocab) + r")\b",
        flags=re.IGNORECASE,
    )

    # Salviamo la lista dei tensori (K, D) per ciascuna teoria per calcolare poi anche il centramento globale
    per_theory_vecs = {l: [] for l in target_layers}
    evaluated = 0

    for r in records:
        q_text = r["question"]
        t_text = r["theory"]

        q_matches = [(m.group(0).lower(), m.start(), m.end()) for m in ent_regex.finditer(q_text)]
        t_matches = [(m.group(0).lower(), m.start(), m.end()) for m in ent_regex.finditer(t_text)]
        unique_ents = sorted(set([e for e, _, _ in q_matches] + [e for e, _, _ in t_matches]))
        if len(unique_ents) < 2:
            continue

        enc = tokenizer(
            q_text,
            t_text,
            truncation=True,
            max_length=384,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        seq_ids = enc.sequence_ids(0)
        offsets = enc["offset_mapping"][0].tolist()
        input_ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
            out = model(
                input_ids=input_ids,
                attention_mask=mask,
                output_hidden_states=True,
            )

        # Trova tutti i token sovrapposti alle occorrenze di ciascuna entità (in question e theory)
        ent_token_indices: Dict[str, List[int]] = {e: [] for e in unique_ents}
        for tok_idx, (sid, (cs, ce)) in enumerate(zip(seq_ids, offsets)):
            if sid is None or cs == ce:
                continue
            matches = q_matches if sid == 0 else t_matches
            for ent_name, ms, me in matches:
                if cs < me and ce > ms:  # sovrapposizione esatta span caratteri
                    ent_token_indices[ent_name].append(tok_idx)

        valid_ents = [e for e in unique_ents if len(ent_token_indices[e]) > 0]
        if len(valid_ents) < 2:
            continue

        for l in target_layers:
            vecs = []
            for e in valid_ents:
                idx_t = torch.tensor(ent_token_indices[e], device=device, dtype=torch.long)
                v = out.hidden_states[l][0, idx_t, :].mean(dim=0).float().cpu()
                vecs.append(v)
            per_theory_vecs[l].append(torch.stack(vecs, dim=0))  # (K, D)

        evaluated += 1
        if evaluated >= max_eval:
            break

    results = {}
    for l in target_layers:
        theories_mats = per_theory_vecs[l]
        all_ent_vecs = torch.cat(theories_mats, dim=0)
        mu = all_ent_vecs.mean(dim=0, keepdim=True)  # (1, D) media globale delle entità

        raw_sims = []
        cent_sims = []
        for mat in theories_mats:
            k = mat.size(0)
            idx_u = torch.triu_indices(k, k, offset=1)

            mat_norm = F.normalize(mat, p=2, dim=-1)
            sim_raw = torch.mm(mat_norm, mat_norm.t())[idx_u[0], idx_u[1]].mean().item()
            raw_sims.append(sim_raw)

            mat_c = F.normalize(mat - mu, p=2, dim=-1)
            sim_cent = torch.mm(mat_c, mat_c.t())[idx_u[0], idx_u[1]].mean().item()
            cent_sims.append(sim_cent)

        results[l] = {
            "raw": float(np.mean(raw_sims)),
            "centered": float(np.mean(cent_sims)),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="distilroberta-base")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    _, train_ds, val_ds, test_id_ds, test_ood_ds = load_or_build_datasets("data_cache", data_seed=42)
    check_delimiters([train_ds.records, val_ds.records, test_id_ds.records, test_ood_ds.records])

    folders = {
        "free_s42": "free_seed42",
        "free_s123": "free_seed123",
        "free_aug_s42": "free_aug_seed42",
        "const025_s42": "constrained_seed42_l3_lam0.25",
        "const05_s42": "constrained_seed42_l3_lam0.5",
        "const05_s123": "constrained_seed123_l3_lam0.5",
        "const10_s42": "constrained_seed42_l3_lam1.0",
    }

    print_unknown_from_eval_json(args.save_dir, folders)

    unk_ood_recs = [r for r in test_ood_ds.records if r["answer"] == "Unknown"]
    entity_vocab = [w.lower() for w in PEOPLE] + [w.replace("_", " ").lower() for w in ANIMALS]

    print(f"\n=== BREAKDOWN FALSI POSITIVI SU OOD UNKNOWN (N={len(unk_ood_recs)}) & ICES (RAW vs CENTERED) ===")
    for label, folder in folders.items():
        ckpt_path = os.path.join(args.save_dir, folder, "best_model.pt")
        if not os.path.exists(ckpt_path):
            continue
        m = load_hf_model(args.model_name, ckpt_path, device)
        fp = eval_unknown_fp_breakdown(m, unk_ood_recs, tokenizer, device)
        ices_id = compute_raw_and_centered_ices(m, test_id_ds.records, tokenizer, device, entity_vocab)
        ices_ood = compute_raw_and_centered_ices(m, test_ood_ds.records, tokenizer, device, entity_vocab)

        print(
            f"\n[{label}]\n"
            f"  OOD Unknown -> Acc={fp['acc_u']:.4f} | FPR_tot={fp['fpr_total']:.4f} "
            f"(->False={fp['fp_to_false']:.4f}, ->True={fp['fp_to_true']:.4f})\n"
            f"  ICES ID     -> L3 raw={ices_id[3]['raw']:.4f} / cent={ices_id[3]['centered']:.4f} | "
            f"L6 raw={ices_id[6]['raw']:.4f} / cent={ices_id[6]['centered']:.4f}\n"
            f"  ICES OOD    -> L3 raw={ices_ood[3]['raw']:.4f} / cent={ices_ood[3]['centered']:.4f} | "
            f"L6 raw={ices_ood[6]['raw']:.4f} / cent={ices_ood[6]['centered']:.4f}"
        )


if __name__ == "__main__":
    main()