import argparse
from collections import defaultdict
import json
import math
import os
import random
import re
import time
from typing import Dict, List, Tuple

from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import (
    ANIMALS,
    ATTRIBUTES,
    ID_TO_LABEL,
    LABEL_MAP,
    PEOPLE,
    VERB_PAIRS,
)
from train import load_or_build_datasets, set_seed


class TextAlphaPermutator:
    """Permutatore biettivo esatto sul testo grezzo di ProofWriter (preserva maiuscole e 'bald eagle')."""

    def __init__(self):
        self.people = [w.lower() for w in PEOPLE]
        # In testo compare come "bald eagle" (con lo spazio)
        self.animals = [w.replace("_", " ").lower() for w in ANIMALS]
        self.attrs = [w.lower() for w in ATTRIBUTES]
        self.verb_pairs = [(b.lower(), s.lower()) for b, s in VERB_PAIRS]

        all_terms = (
            self.animals  # 'bald eagle' prima dei singoli se servisse
            + self.people
            + self.attrs
            + [w for p in self.verb_pairs for w in p]
        )
        # Ordina per lunghezza decrescente così "bald eagle" matcha sempre intero
        all_terms = sorted(set(all_terms), key=len, reverse=True)
        self.pattern = re.compile(
            r"\b(" + "|".join(re.escape(t) for t in all_terms) + r")\b",
            flags=re.IGNORECASE,
        )

    def sample_word_map(self, rng: random.Random) -> Dict[str, str]:
        wmap = {}

        p_shuf = self.people.copy()
        rng.shuffle(p_shuf)
        for s, d in zip(self.people, p_shuf):
            wmap[s] = d

        a_shuf = self.animals.copy()
        rng.shuffle(a_shuf)
        for s, d in zip(self.animals, a_shuf):
            wmap[s] = d

        at_shuf = self.attrs.copy()
        rng.shuffle(at_shuf)
        for s, d in zip(self.attrs, at_shuf):
            wmap[s] = d

        v_shuf = self.verb_pairs.copy()
        rng.shuffle(v_shuf)
        for (bs, ss), (bd, sd) in zip(self.verb_pairs, v_shuf):
            wmap[bs] = bd
            wmap[ss] = sd

        return wmap

    def permute_pair(
        self, question: str, theory: str, rng: random.Random
    ) -> Tuple[str, str]:
        wmap = self.sample_word_map(rng)

        def repl(match: re.Match) -> str:
            orig = match.group(0)
            low = orig.lower()
            mapped = wmap.get(low, low)
            if orig[0].isupper():
                return mapped[0].upper() + mapped[1:]
            return mapped

        return self.pattern.sub(repl, question), self.pattern.sub(repl, theory)


def load_or_build_natlang(
    cache_dir: str,
    train_theories: set,
    max_samples: int = 3000,
    seed: int = 42,
) -> List[dict]:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"proofwriter_natlang_seed{seed}.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=False)

    print("Estrazione split OOD NatLang...")
    raw_test = load_dataset("tasksource/proofwriter", split="test")
    pool = [
        r
        for r in raw_test
        if "NatLang" in str(r["config"]) and r["theory"] not in train_theories
    ]
    rng = random.Random(seed)
    by_cls = defaultdict(list)
    for r in pool:
        by_cls[r["answer"]].append(r)

    selected = []
    per_cls = max_samples // 3
    for cls_name in ("False", "True", "Unknown"):
        cls_pool = by_cls[cls_name]
        rng.shuffle(cls_pool)
        selected.extend(cls_pool[:per_cls])
    rng.shuffle(selected)

    torch.save(selected, cache_path)
    return selected


def compute_sentence_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    dot_id: int,
    cls_id: int,
    sep_id: int,
) -> torch.Tensor:
    """Assegna un indice di frase (0=question, 1..M=frasi theory) a ogni token valido, -1 altrove."""
    is_dot = (input_ids == dot_id).long()
    sent_ids = torch.cumsum(is_dot, dim=1) - is_dot
    valid_tok = (
        (attention_mask == 1) & (input_ids != cls_id) & (input_ids != sep_id)
    )
    return torch.where(valid_tok, sent_ids, torch.full_like(sent_ids, -1))


def pool_sentences(
    hidden: torch.Tensor, sent_ids: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Media vettorizzata degli stati nascosti per ciascuna frase: (B, max_S, D), (B, max_S)."""
    B, T, D = hidden.shape
    max_s = int(sent_ids.max().item()) + 1
    if max_s <= 0:
        return torch.zeros(B, 1, D, device=hidden.device), torch.zeros(
            B, 1, dtype=torch.bool, device=hidden.device
        )

    valid = sent_ids >= 0
    b_idx, t_idx = valid.nonzero(as_tuple=True)
    s_idx = sent_ids[b_idx, t_idx]
    flat_group = b_idx * max_s + s_idx

    sums = torch.zeros(
        B * max_s, D, device=hidden.device, dtype=torch.float32
    )
    cnts = torch.zeros(B * max_s, 1, device=hidden.device, dtype=torch.float32)

    sums.index_add_(0, flat_group, hidden[b_idx, t_idx].float())
    cnts.index_add_(
        0,
        flat_group,
        torch.ones(flat_group.size(0), 1, device=hidden.device, dtype=torch.float32),
    )

    reprs = (sums / cnts.clamp(min=1.0)).view(B, max_s, D)
    valid_s = cnts.view(B, max_s) > 0
    return reprs, valid_s


def compute_sentence_invariance_loss(
    h_x: torch.Tensor,
    sent_ids_x: torch.Tensor,
    h_px: torch.Tensor,
    sent_ids_px: torch.Tensor,
) -> torch.Tensor:
    repr_x, valid_x = pool_sentences(h_x, sent_ids_x)
    repr_px, valid_px = pool_sentences(h_px, sent_ids_px)

    s = min(repr_x.size(1), repr_px.size(1))
    mask = valid_x[:, :s] & valid_px[:, :s]
    if not mask.any():
        return torch.tensor(0.0, device=h_x.device)

    vx = repr_x[:, :s, :][mask]
    vpx = repr_px[:, :s, :][mask]
    mu = torch.cat([vx, vpx], dim=0).mean(dim=0, keepdim=True)

    cos_centered = F.cosine_similarity(vx - mu, vpx - mu, dim=-1)
    return (1.0 - cos_centered).mean()


@torch.no_grad()
def evaluate_hf(
    model,
    records: List[dict],
    hf_tok,
    permutator: TextAlphaPermutator,
    device: torch.device,
    dot_id: int,
    inv_layer: int = 3,
    batch_size: int = 64,
    n_perm_flips: int = 1,
    eval_seed: int = 999,
) -> Dict[str, object]:
    model.eval()
    rng = random.Random(eval_seed)

    total = 0
    correct = 0
    total_flips = 0
    total_perm_checks = 0
    inv_cos_sum = 0.0
    inv_batches = 0

    correct_by_qdep = defaultdict(int)
    total_by_qdep = defaultdict(int)
    correct_tf_by_qdep = defaultdict(int)
    total_tf_by_qdep = defaultdict(int)
    correct_by_cls = defaultdict(int)
    total_by_cls = defaultdict(int)

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        qs = [r["question"] for r in batch]
        ths = [r["theory"] for r in batch]
        labels = torch.tensor(
            [LABEL_MAP[r["answer"]] for r in batch], dtype=torch.long, device=device
        )
        qdeps = [int(r["QDep"]) for r in batch]

        enc = hf_tok(
            qs, ths, padding=True, truncation=True, max_length=384, return_tensors="pt"
        )
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
            out = model(
                input_ids=ids,
                attention_mask=mask,
                output_hidden_states=(n_perm_flips > 0),
            )
        preds = out.logits.argmax(dim=-1)
        matches = preds == labels

        total += labels.size(0)
        correct += int(matches.sum().item())

        for j in range(labels.size(0)):
            qd = qdeps[j]
            lbl = int(labels[j].item())
            ok = int(matches[j].item())
            total_by_qdep[qd] += 1
            correct_by_qdep[qd] += ok
            total_by_cls[lbl] += 1
            correct_by_cls[lbl] += ok
            if lbl in (0, 1):
                total_tf_by_qdep[qd] += 1
                correct_tf_by_qdep[qd] += ok

        if n_perm_flips > 0:
            sent_x = compute_sentence_ids(
                ids, mask, dot_id, hf_tok.cls_token_id, hf_tok.sep_token_id
            )
            h_x = out.hidden_states[inv_layer]

            for f_idx in range(n_perm_flips):
                p_pairs = [
                    permutator.permute_pair(q, t, rng) for q, t in zip(qs, ths)
                ]
                p_qs = [p[0] for p in p_pairs]
                p_ths = [p[1] for p in p_pairs]
                penc = hf_tok(
                    p_qs,
                    p_ths,
                    padding=True,
                    truncation=True,
                    max_length=384,
                    return_tensors="pt",
                )
                pids = penc["input_ids"].to(device)
                pmask = penc["attention_mask"].to(device)

                with torch.amp.autocast(
                    device.type, enabled=(device.type == "cuda")
                ):
                    pout = model(
                        input_ids=pids,
                        attention_mask=pmask,
                        output_hidden_states=(f_idx == 0),
                    )
                ppreds = pout.logits.argmax(dim=-1)
                total_flips += int((ppreds != preds).sum().item())
                total_perm_checks += labels.size(0)

                if f_idx == 0:
                    sent_px = compute_sentence_ids(
                        pids,
                        pmask,
                        dot_id,
                        hf_tok.cls_token_id,
                        hf_tok.sep_token_id,
                    )
                    inv_loss = compute_sentence_invariance_loss(
                        h_x, sent_x, pout.hidden_states[inv_layer], sent_px
                    )
                    inv_cos_sum += float(1.0 - inv_loss.item())
                    inv_batches += 1

    return {
        "accuracy": correct / max(total, 1),
        "flip_rate": (
            total_flips / max(total_perm_checks, 1) if n_perm_flips > 0 else None
        ),
        "layer_inv_cos": (inv_cos_sum / inv_batches if inv_batches > 0 else None),
        "n_samples": total,
        "acc_by_qdep": {
            str(k): correct_by_qdep[k] / total_by_qdep[k]
            for k in sorted(total_by_qdep.keys())
        },
        "acc_true_false_by_qdep": {
            str(k): correct_tf_by_qdep[k] / total_tf_by_qdep[k]
            for k in sorted(total_tf_by_qdep.keys())
        },
        "acc_by_class": {
            ID_TO_LABEL[k]: correct_by_cls[k] / total_by_cls[k]
            for k in sorted(total_by_cls.keys())
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["free", "free_aug", "constrained"],
    )
    parser.add_argument("--model_name", type=str, default="distilroberta-base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lambda_inv", type=float, default=1.0)
    parser.add_argument("--inv_layer", type=int, default=3)
    parser.add_argument("--cache_dir", type=str, default="data_cache")
    parser.add_argument("--save_dir", type=str, default="results_hf")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    run_name = f"{args.mode}_seed{args.seed}"
    if args.mode == "constrained":
        run_name += f"_l{args.inv_layer}_lam{args.lambda_inv}"
    if args.debug:
        run_name += "_debug"

    out_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"=== RUN HF: {run_name} | model={args.model_name} | device={device} ===")

    _, train_ds, val_ds, test_id_ds, test_ood_ds = load_or_build_datasets(
        args.cache_dir, data_seed=args.data_seed
    )
    train_records = train_ds.records
    val_records = val_ds.records
    test_id_records = test_id_ds.records
    test_ood_records = test_ood_ds.records

    if args.debug:
        args.epochs = 2
        args.batch_size = min(args.batch_size, 8)
        train_records = train_records[:64]
        val_records = val_records[:32]
        test_id_records = test_id_records[:32]
        test_ood_records = test_ood_records[:32]
        natlang_records = val_records[:32]
    else:
        train_theories = set(r["theory"] for r in train_records)
        natlang_records = load_or_build_natlang(
            args.cache_dir, train_theories, max_samples=3000, seed=args.data_seed
        )

    # Seed identico PRIMA del caricamento modello (garantisce stessa testa classifier iniziale!)
    set_seed(args.seed)
    perm_rng = random.Random(args.seed + 1000)
    permutator = TextAlphaPermutator()

    hf_tok = AutoTokenizer.from_pretrained(args.model_name)
    dot_ids = hf_tok.encode(".", add_special_tokens=False)
    assert len(dot_ids) == 1, f"Atteso 1 token per '.', trovato {dot_ids}"
    dot_id = dot_ids[0]

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=3
    ).to(device)

    def collate_records(batch: List[dict]) -> Dict[str, object]:
        qs = [r["question"] for r in batch]
        ths = [r["theory"] for r in batch]
        labels = torch.tensor(
            [LABEL_MAP[r["answer"]] for r in batch], dtype=torch.long
        )
        return {"qs": qs, "ths": ths, "labels": labels}

    train_loader = DataLoader(
        train_records,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_records,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, int(0.06 * total_steps))

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_val_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        sum_ce = 0.0
        sum_inv = 0.0
        sum_correct = 0
        sum_samples = 0

        for batch in train_loader:
            qs, ths = batch["qs"], batch["ths"]
            labels = batch["labels"].to(device)
            B = labels.size(0)

            optimizer.zero_grad(set_to_none=True)

            if args.mode == "free":
                enc = hf_tok(
                    qs,
                    ths,
                    padding=True,
                    truncation=True,
                    max_length=384,
                    return_tensors="pt",
                )
                ids = enc["input_ids"].to(device)
                mask = enc["attention_mask"].to(device)

                with torch.amp.autocast(device.type, enabled=use_amp):
                    out = model(input_ids=ids, attention_mask=mask)
                    logits_x = out.logits
                    loss_ce = F.cross_entropy(logits_x, labels)
                    loss_inv = torch.tensor(0.0, device=device)
                    loss = loss_ce
            else:
                p_pairs = [
                    permutator.permute_pair(q, t, perm_rng)
                    for q, t in zip(qs, ths)
                ]
                p_qs = [p[0] for p in p_pairs]
                p_ths = [p[1] for p in p_pairs]

                # Tokenizziamo insieme così il padding di batch combacia se serve, oppure separati
                enc = hf_tok(
                    qs + p_qs,
                    ths + p_ths,
                    padding=True,
                    truncation=True,
                    max_length=384,
                    return_tensors="pt",
                )
                ids_all = enc["input_ids"].to(device)
                mask_all = enc["attention_mask"].to(device)

                need_h = args.mode == "constrained"
                with torch.amp.autocast(device.type, enabled=use_amp):
                    out = model(
                        input_ids=ids_all,
                        attention_mask=mask_all,
                        output_hidden_states=need_h,
                    )
                    logits_x, logits_px = out.logits.split(B, dim=0)
                    loss_ce = 0.5 * (
                        F.cross_entropy(logits_x, labels)
                        + F.cross_entropy(logits_px, labels)
                    )

                    if args.mode == "constrained":
                        ids_x, ids_px = ids_all.split(B, dim=0)
                        mask_x, mask_px = mask_all.split(B, dim=0)
                        sent_x = compute_sentence_ids(
                            ids_x,
                            mask_x,
                            dot_id,
                            hf_tok.cls_token_id,
                            hf_tok.sep_token_id,
                        )
                        sent_px = compute_sentence_ids(
                            ids_px,
                            mask_px,
                            dot_id,
                            hf_tok.cls_token_id,
                            hf_tok.sep_token_id,
                        )
                        h_x, h_px = out.hidden_states[args.inv_layer].split(
                            B, dim=0
                        )
                        loss_inv = compute_sentence_invariance_loss(
                            h_x, sent_x, h_px, sent_px
                        )
                        loss = loss_ce + args.lambda_inv * loss_inv
                    else:
                        loss_inv = torch.tensor(0.0, device=device)
                        loss = loss_ce

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            sum_ce += float(loss_ce.item()) * B
            sum_inv += float(loss_inv.item()) * B
            sum_correct += int((logits_x.argmax(dim=-1) == labels).sum().item())
            sum_samples += B

        train_ce = sum_ce / sum_samples
        train_inv = sum_inv / sum_samples
        train_acc = sum_correct / sum_samples

        val_m = evaluate_hf(
            model,
            val_records,
            hf_tok,
            permutator,
            device,
            dot_id=dot_id,
            inv_layer=args.inv_layer,
            batch_size=16 if args.debug else 64,
            n_perm_flips=1,
            eval_seed=args.seed + epoch,
        )
        dt = time.time() - t0

        row = {
            "epoch": epoch,
            "train_ce": train_ce,
            "train_inv": train_inv,
            "train_acc": train_acc,
            "val_acc": val_m["accuracy"],
            "val_acc_cls": val_m["acc_by_class"],
            "val_flip_rate": val_m["flip_rate"],
            "val_layer_inv_cos": val_m["layer_inv_cos"],
            "time_sec": dt,
        }
        history.append(row)

        cls_str = (
            f"T={val_m['acc_by_class']['True']:.2f}/"
            f"F={val_m['acc_by_class']['False']:.2f}/"
            f"U={val_m['acc_by_class']['Unknown']:.2f}"
        )
        print(
            f"Ep {epoch:02d}/{args.epochs:02d} ({dt:.1f}s) | "
            f"ce={train_ce:.4f} | inv={train_inv:.4f} | tr_acc={train_acc:.3f} | "
            f"val_acc={val_m['accuracy']:.3f} ({cls_str}) | "
            f"flip={val_m['flip_rate']:.3f} | l{args.inv_layer}_cos={val_m['layer_inv_cos']:.3f}"
        )

        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch},
                os.path.join(out_dir, "best_model.pt"),
            )

    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": args.epochs},
        os.path.join(out_dir, "last_model.pt"),
    )

    eval_all = {}
    eval_bs = 16 if args.debug else 64
    for ckpt_tag in ("best_model", "last_model"):
        ckpt = torch.load(
            os.path.join(out_dir, f"{ckpt_tag}.pt"),
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(ckpt["model_state_dict"])

        res_id = evaluate_hf(
            model,
            test_id_records,
            hf_tok,
            permutator,
            device,
            dot_id=dot_id,
            inv_layer=args.inv_layer,
            batch_size=eval_bs,
            n_perm_flips=3,
            eval_seed=101,
        )
        res_ood = evaluate_hf(
            model,
            test_ood_records,
            hf_tok,
            permutator,
            device,
            dot_id=dot_id,
            inv_layer=args.inv_layer,
            batch_size=eval_bs,
            n_perm_flips=3,
            eval_seed=202,
        )
        res_nat = evaluate_hf(
            model,
            natlang_records,
            hf_tok,
            permutator,
            device,
            dot_id=dot_id,
            inv_layer=args.inv_layer,
            batch_size=eval_bs,
            n_perm_flips=0,
            eval_seed=303,
        )
        eval_all[ckpt_tag] = {
            "epoch": ckpt["epoch"],
            "test_id": res_id,
            "test_ood_depth": res_ood,
            "test_ood_natlang": res_nat,
        }

    with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
        json.dump(eval_all, f, indent=2)

    print("\n=== RISULTATI FINALI (best_model) ===")
    for label, key in [
        ("Test ID      ", "test_id"),
        ("Test OOD Dep ", "test_ood_depth"),
        ("Test NatLang ", "test_ood_natlang"),
    ]:
        r = eval_all["best_model"][key]
        print(
            f"{label} | Acc={r['accuracy']:.4f} | Flip={r['flip_rate']} | "
            f"L{args.inv_layer}_cos={r['layer_inv_cos']} | Cls={r['acc_by_class']}"
        )
    print("  ID per QDep:", eval_all["best_model"]["test_id"]["acc_by_qdep"])
    print(
        "  OOD True/False per QDep:",
        eval_all["best_model"]["test_ood_depth"]["acc_true_false_by_qdep"],
    )


if __name__ == "__main__":
    main()