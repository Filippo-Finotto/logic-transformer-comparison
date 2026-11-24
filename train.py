import argparse
from collections import defaultdict
import json
import math
import os
import random
import time
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, Subset

from data import (
    AlphaPermutator,
    ID_TO_LABEL,
    build_datasets,
)
from model import LogicTransformer, count_parameters


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LengthBucketBatchSampler(Sampler[List[int]]):
    """Raggruppa esempi di lunghezza simile per ridurre gli sprechi sul padding mantenendo stocasticità."""

    def __init__(self, lengths: List[int], batch_size: int, seed: int = 42):
        self.lengths = lengths
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        rng.shuffle(indices)

        # Ordina per lunghezza dentro mega-blocchi da 25 batch
        pool_size = self.batch_size * 25
        batches = []
        for i in range(0, len(indices), pool_size):
            pool = indices[i : i + pool_size]
            pool.sort(key=lambda idx: self.lengths[idx])
            for j in range(0, len(pool), self.batch_size):
                batches.append(pool[j : j + self.batch_size])

        rng.shuffle(batches)
        for b in batches:
            yield b

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def get_dataset_lengths(ds) -> List[int]:
    if isinstance(ds, Subset):
        masks = ds.dataset.attention_mask[ds.indices]
    else:
        masks = ds.attention_mask
    return masks.sum(dim=1).tolist()


def dynamic_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    qdeps = torch.stack([b["qdep"] for b in batch], dim=0)

    max_real_len = int(attention_mask.sum(dim=1).max().item())
    max_real_len = max(max_real_len, 1)

    return {
        "input_ids": input_ids[:, :max_real_len],
        "attention_mask": attention_mask[:, :max_real_len],
        "label": labels,
        "qdep": qdeps,
    }


def compute_cosine_invariance_loss(
    h_x: torch.Tensor, h_px: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    cos_sim = F.cosine_similarity(h_x.float(), h_px.float(), dim=-1)
    mask = attention_mask.float()
    return ((1.0 - cos_sim) * mask).sum() / mask.sum().clamp(min=1.0)


def load_or_build_datasets(cache_dir: str, data_seed: int = 42):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"proofwriter_qfirst_seed{data_seed}.pt")
    if os.path.exists(cache_path):
        print(f"Caricamento dataset da cache: {cache_path}")
        saved = torch.load(cache_path, weights_only=False)
        return (
            saved["tokenizer"],
            saved["train_ds"],
            saved["val_ds"],
            saved["test_id_ds"],
            saved["test_ood_ds"],
        )

    tok, train_ds, val_ds, test_id_ds, test_ood_ds = build_datasets(
        seed=data_seed
    )
    torch.save(
        {
            "tokenizer": tok,
            "train_ds": train_ds,
            "val_ds": val_ds,
            "test_id_ds": test_id_ds,
            "test_ood_ds": test_ood_ds,
        },
        cache_path,
    )
    print(f"Dataset salvato in cache: {cache_path}")
    return tok, train_ds, val_ds, test_id_ds, test_ood_ds


@torch.no_grad()
def evaluate(
    model: LogicTransformer,
    loader: DataLoader,
    permutator: AlphaPermutator,
    device: torch.device,
    n_perm_flips: int = 1,
    eval_seed: int = 999,
) -> Dict[str, object]:
    model.eval()
    rng = random.Random(eval_seed)

    total = 0
    correct = 0
    total_flips = 0
    total_perm_checks = 0

    correct_by_qdep = defaultdict(int)
    total_by_qdep = defaultdict(int)

    correct_tf_by_qdep = defaultdict(int)
    total_tf_by_qdep = defaultdict(int)

    correct_by_cls = defaultdict(int)
    total_by_cls = defaultdict(int)

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        qdeps = batch["qdep"].to(device)

        logits = model(input_ids, mask)["logits"]
        preds = logits.argmax(dim=-1)

        matches = preds == labels
        total += labels.size(0)
        correct += int(matches.sum().item())

        for i in range(labels.size(0)):
            qd = int(qdeps[i].item())
            lbl = int(labels[i].item())
            ok = int(matches[i].item())

            total_by_qdep[qd] += 1
            correct_by_qdep[qd] += ok

            total_by_cls[lbl] += 1
            correct_by_cls[lbl] += ok

            if lbl in (0, 1):
                total_tf_by_qdep[qd] += 1
                correct_tf_by_qdep[qd] += ok

        for _ in range(n_perm_flips):
            perm_ids = permutator.permute_batch(input_ids, rng)
            perm_preds = model(perm_ids, mask)["logits"].argmax(dim=-1)
            total_flips += int((perm_preds != preds).sum().item())
            total_perm_checks += labels.size(0)

    acc_by_qdep = {
        str(k): correct_by_qdep[k] / total_by_qdep[k]
        for k in sorted(total_by_qdep.keys())
    }
    acc_tf_by_qdep = {
        str(k): correct_tf_by_qdep[k] / total_tf_by_qdep[k]
        for k in sorted(total_tf_by_qdep.keys())
    }
    acc_by_cls = {
        ID_TO_LABEL[k]: correct_by_cls[k] / total_by_cls[k]
        for k in sorted(total_by_cls.keys())
    }

    return {
        "accuracy": correct / max(total, 1),
        "flip_rate": total_flips / max(total_perm_checks, 1),
        "n_samples": total,
        "acc_by_qdep": acc_by_qdep,
        "acc_true_false_by_qdep": acc_tf_by_qdep,
        "acc_by_class": acc_by_cls,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["free", "free_aug", "constrained"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lambda_inv", type=float, default=1.0)
    parser.add_argument("--inv_layer", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cache_dir", type=str, default="data_cache")
    parser.add_argument("--save_dir", type=str, default="results")
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

    print(f"=== RUN: {run_name} | device={device} | amp={use_amp} ===")

    tok, train_ds, val_ds, test_id_ds, test_ood_ds = load_or_build_datasets(
        args.cache_dir, data_seed=args.data_seed
    )

    if args.debug:
        args.epochs = 2
        args.batch_size = min(args.batch_size, 8)
        train_ds = Subset(train_ds, list(range(64)))
        val_ds = Subset(val_ds, list(range(32)))
        test_id_ds = Subset(test_id_ds, list(range(32)))
        test_ood_ds = Subset(test_ood_ds, list(range(32)))

    set_seed(args.seed)
    perm_rng = random.Random(args.seed + 1000)
    permutator = AlphaPermutator(tok)

    train_sampler = LengthBucketBatchSampler(
        get_dataset_lengths(train_ds), batch_size=args.batch_size, seed=args.seed
    )
    eval_bs = 16 if args.debug else 128

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=dynamic_collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=eval_bs, shuffle=False, collate_fn=dynamic_collate_fn
    )
    test_id_loader = DataLoader(
        test_id_ds, batch_size=eval_bs, shuffle=False, collate_fn=dynamic_collate_fn
    )
    test_ood_loader = DataLoader(
        test_ood_ds, batch_size=eval_bs, shuffle=False, collate_fn=dynamic_collate_fn
    )

    model = LogicTransformer(
        vocab_size=tok.vocab_size,
        max_len=320,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=4 * args.d_model,
        num_classes=3,
        dropout=args.dropout,
    ).to(device)
    print(f"Parametri modello: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, int(0.05 * total_steps))
    min_lr_ratio = 0.05  # Non scende mai sotto il 5% di lr_max

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_val_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_sampler.set_epoch(epoch)
        model.train()

        sum_ce = 0.0
        sum_inv = 0.0
        sum_correct = 0
        sum_samples = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device.type, enabled=use_amp):
                if args.mode == "free":
                    out_x = model(input_ids, mask)
                    logits_x = out_x["logits"]
                    loss_ce = F.cross_entropy(logits_x, labels)
                    loss_inv = torch.tensor(0.0, device=device)
                    loss = loss_ce
                else:
                    perm_ids = permutator.permute_batch(input_ids, perm_rng)
                    need_h = args.mode == "constrained"

                    both_ids = torch.cat([input_ids, perm_ids], dim=0)
                    both_mask = torch.cat([mask, mask], dim=0)
                    out_both = model(
                        both_ids, both_mask, return_hidden_states=need_h
                    )

                    B = input_ids.size(0)
                    logits_x, logits_px = out_both["logits"].split(B, dim=0)
                    loss_ce = 0.5 * (
                        F.cross_entropy(logits_x, labels)
                        + F.cross_entropy(logits_px, labels)
                    )

                    if args.mode == "constrained":
                        h_layer = out_both["hidden_states"][args.inv_layer]
                        h_x, h_px = h_layer.split(B, dim=0)
                        loss_inv = compute_cosine_invariance_loss(h_x, h_px, mask)
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

            B = labels.size(0)
            sum_ce += float(loss_ce.item()) * B
            sum_inv += float(loss_inv.item()) * B
            sum_correct += int((logits_x.argmax(dim=-1) == labels).sum().item())
            sum_samples += B

        train_ce = sum_ce / sum_samples
        train_inv = sum_inv / sum_samples
        train_acc = sum_correct / sum_samples

        val_metrics = evaluate(
            model,
            val_loader,
            permutator,
            device,
            n_perm_flips=1,
            eval_seed=args.seed + epoch,
        )
        dt = time.time() - t0

        row_log = {
            "epoch": epoch,
            "train_ce": train_ce,
            "train_inv": train_inv,
            "train_acc": train_acc,
            "val_acc": val_metrics["accuracy"],
            "val_acc_cls": val_metrics["acc_by_class"],
            "val_flip_rate": val_metrics["flip_rate"],
            "time_sec": dt,
        }
        history.append(row_log)

        cls_str = (
            f"T={val_metrics['acc_by_class']['True']:.2f}/"
            f"F={val_metrics['acc_by_class']['False']:.2f}/"
            f"U={val_metrics['acc_by_class']['Unknown']:.2f}"
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} ({dt:.1f}s) | "
            f"ce={train_ce:.4f} | inv={train_inv:.4f} | "
            f"tr_acc={train_acc:.3f} | val_acc={val_metrics['accuracy']:.3f} ({cls_str}) | "
            f"flip={val_metrics['flip_rate']:.3f}"
        )

        # Salva history.json ad ogni epoca (sicuro contro disconnessioni!)
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch},
                os.path.join(out_dir, "best_model.pt"),
            )

    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": args.epochs},
        os.path.join(out_dir, "last_model.pt"),
    )

    eval_all = {}
    for ckpt_tag in ("best_model", "last_model"):
        ckpt = torch.load(
            os.path.join(out_dir, f"{ckpt_tag}.pt"),
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(ckpt["model_state_dict"])

        res_id = evaluate(
            model, test_id_loader, permutator, device, n_perm_flips=3, eval_seed=101
        )
        res_ood = evaluate(
            model, test_ood_loader, permutator, device, n_perm_flips=3, eval_seed=202
        )
        eval_all[ckpt_tag] = {
            "epoch": ckpt["epoch"],
            "test_id": res_id,
            "test_ood": res_ood,
        }

    with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
        json.dump(eval_all, f, indent=2)

    print("\n=== RISULTATI FINALI (best_model) ===")
    b_id = eval_all["best_model"]["test_id"]
    b_ood = eval_all["best_model"]["test_ood"]
    print(f"Test ID  | Acc={b_id['accuracy']:.4f} | Flip={b_id['flip_rate']:.4f} | Cls={b_id['acc_by_class']}")
    print(f"  per QDep: {b_id['acc_by_qdep']}")
    print(f"Test OOD | Acc={b_ood['accuracy']:.4f} | Flip={b_ood['flip_rate']:.4f} | Cls={b_ood['acc_by_class']}")
    print(f"  True/False per QDep: {b_ood['acc_true_false_by_qdep']}")


if __name__ == "__main__":
    main()