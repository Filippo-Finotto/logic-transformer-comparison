import argparse
import os
import re
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from analyze_hf import find_models_in_dir, load_hf_model
from data import ANIMALS, PEOPLE
from train import load_or_build_datasets, set_seed


def extract_entity_representations(
    model,
    records: List[dict],
    tokenizer,
    device: torch.device,
    entity_vocab: List[str],
    target_layers: List[int] = [3, 6],
    max_eval: int = 500,
) -> Dict[int, float]:
    model.eval()
    layer_scores = {l: [] for l in target_layers}

    # Regex per trovare entità intere
    ent_regex = re.compile(
        r"\b(" + "|".join(re.escape(e) for e in entity_vocab) + r")\b",
        flags=re.IGNORECASE,
    )

    evaluated = 0
    for r in records:
        theory = r["theory"].lower()
        found_entities = sorted(list(set(ent_regex.findall(theory))))
        if len(found_entities) < 2:
            continue

        enc = tokenizer(
            r["question"],
            r["theory"],
            padding=True,
            truncation=True,
            max_length=384,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            with torch.amp.autocast(
                device.type, enabled=(device.type == "cuda")
            ):
                out = model(
                    input_ids=input_ids,
                    attention_mask=mask,
                    output_hidden_states=True,
                )

        # Mappatura token -> posizioni per ogni entità
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        ent_vectors = {l: [] for l in target_layers}

        for ent in found_entities:
            # Token ID dell'entità preceduta da spazio (convenzione BPE di RoBERTa)
            ent_tok = tokenizer.tokenize(" " + ent)
            ent_len = len(ent_tok)

            # Cerca la sequenza di token nel testo
            matched_indices = []
            for idx in range(len(tokens) - ent_len + 1):
                if tokens[idx : idx + ent_len] == ent_tok:
                    matched_indices.extend(range(idx, idx + ent_len))

            if matched_indices:
                matched_t = torch.tensor(
                    matched_indices, device=device, dtype=torch.long
                )
                for l in target_layers:
                    vec = (
                        out.hidden_states[l][0, matched_t, :]
                        .mean(dim=0)
                        .float()
                    )
                    ent_vectors[l].append(vec)

        # Calcola similarità a coppie per le entità di questo contesto
        for l in target_layers:
            vecs = ent_vectors[l]
            if len(vecs) >= 2:
                V = torch.stack(vecs, dim=0)  # (K, D)
                V_norm = F.normalize(V, p=2, dim=-1)
                cos_sim_mat = torch.mm(V_norm, V_norm.t())
                # Estrai triangolare superiore (escludendo diagonale auto-similarità)
                idx_upper = torch.triu_indices(len(vecs), len(vecs), offset=1)
                avg_sim = cos_sim_mat[idx_upper[0], idx_upper[1]].mean().item()
                layer_scores[l].append(avg_sim)

        evaluated += 1
        if evaluated >= max_eval:
            break

    return {l: float(np.mean(layer_scores[l])) for l in target_layers}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="distilroberta-base")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_map = find_models_in_dir(args.save_dir, seed=args.seed)
    print("Modelli rilevati:", ckpt_map)

    _, _, _, test_id_ds, test_ood_ds = load_or_build_datasets(
        "data_cache", data_seed=42
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    entity_vocab = [w.lower() for w in PEOPLE] + [
        w.replace("_", " ").lower() for w in ANIMALS
    ]

    print("\n=== INTRA-CONTEXT ENTITY SIMILARITY (ICES) ===")
    print("Test ID (400 teorie multi-entità):")
    for name, path in ckpt_map.items():
        m = load_hf_model(args.model_name, path, device)
        res = extract_entity_representations(
            m,
            test_id_ds.records,
            tokenizer,
            device,
            entity_vocab,
            target_layers=[3, 6],
            max_eval=400,
        )
        print(
            f"  {name:20s} -> Layer 3 ICES: {res[3]:.4f} | Layer 6 ICES: {res[6]:.4f}"
        )

    print("\nTest OOD Depth (400 teorie multi-entità):")
    for name, path in ckpt_map.items():
        m = load_hf_model(args.model_name, path, device)
        res = extract_entity_representations(
            m,
            test_ood_ds.records,
            tokenizer,
            device,
            entity_vocab,
            target_layers=[3, 6],
            max_eval=400,
        )
        print(
            f"  {name:20s} -> Layer 3 ICES: {res[3]:.4f} | Layer 6 ICES: {res[6]:.4f}"
        )


if __name__ == "__main__":
    main()