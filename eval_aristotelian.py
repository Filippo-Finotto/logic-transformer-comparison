import argparse
import inspect
import json
import math
import os
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from analyze_hf import load_hf_model
from train import load_or_build_datasets, set_seed
from train_hf import compute_sentence_ids, pool_sentences

try:
    from calibrated_similarity import calibrate, calibrate_layers
    HAS_CALIBRATED_SIM = True
except ImportError:
    HAS_CALIBRATED_SIM = False


# =====================================================================
# 1. METRICHE DI SIMILARITA' (con centramento anti-anisotropia)
# =====================================================================
def cka_linear(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    ytx = Yc.t() @ Xc
    xtx = Xc.t() @ Xc
    yty = Yc.t() @ Yc
    num = (ytx * ytx).sum()
    den = torch.sqrt((xtx * xtx).sum() * (yty * yty).sum()).clamp(min=1e-12)
    return num / den


def make_mknn_fn(k: int = 10) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def mknn(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        # Centramento anti-anisotropia prima della normalizzazione coseno
        Xn = F.normalize(X - X.mean(dim=0, keepdim=True), p=2, dim=-1)
        Yn = F.normalize(Y - Y.mean(dim=0, keepdim=True), p=2, dim=-1)
        Sx = Xn @ Xn.t()
        Sy = Yn @ Yn.t()
        Sx.fill_diagonal_(-1e9)
        Sy.fill_diagonal_(-1e9)
        nx = Sx.topk(k, dim=-1).indices
        ny = Sy.topk(k, dim=-1).indices
        Ax = torch.zeros_like(Sx, dtype=torch.bool).scatter_(1, nx, True)
        Ay = torch.zeros_like(Sy, dtype=torch.bool).scatter_(1, ny, True)
        return (Ax & Ay).sum(dim=1).float().mean() / float(k)

    return mknn


def make_cknna_fn(k: int = 10) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def cknna(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        # Centramento prima del masking (evita overflow > 1 con feature anisotropiche)
        Xc = X - X.mean(dim=0, keepdim=True)
        Yc = Y - Y.mean(dim=0, keepdim=True)
        Xn = F.normalize(Xc, p=2, dim=-1)
        Yn = F.normalize(Yc, p=2, dim=-1)
        Sx = Xn @ Xn.t()
        Sy = Yn @ Yn.t()
        Sx.fill_diagonal_(-1e9)
        Sy.fill_diagonal_(-1e9)
        nx = Sx.topk(k, dim=-1).indices
        ny = Sy.topk(k, dim=-1).indices
        Ax = torch.zeros_like(Sx, dtype=torch.bool).scatter_(1, nx, True)
        Ay = torch.zeros_like(Sy, dtype=torch.bool).scatter_(1, ny, True)
        M = (Ax & Ay).float()

        K_c = Xc @ Xc.t()
        L_c = Yc @ Yc.t()
        hsic_xy = (K_c * L_c * M).sum()
        hsic_xx = (K_c * K_c * M).sum()
        hsic_yy = (L_c * L_c * M).sum()
        return hsic_xy / torch.sqrt(hsic_xx * hsic_yy).clamp(min=1e-12)

    return cknna


# =====================================================================
# 2. CALIBRAZIONE NULL-PERMUTATION (Gröger et al., 2026)
# =====================================================================
def calibrate_layer_matrix(
    feats_a: List[torch.Tensor],
    feats_b: List[torch.Tensor],
    metric_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    n_perms: int = 200,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Usa `calibrated_similarity.calibrate_layers` se compatibile con la firma attesa,
    altrimenti esegue l'Algoritmo 2 (aggregation-aware calibration) di Gröger et al. (2026).
    Restituisce: (raw_7x7, cal_7x7, tau_agg, max_cal).
    """
    L = len(feats_a)
    N = feats_a[0].size(0)
    raw_mat = np.zeros((L, L), dtype=np.float64)
    for i in range(L):
        for j in range(L):
            raw_mat[i, j] = float(metric_fn(feats_a[i], feats_b[j]).item())

    # Tentativo diretto tramite pacchetto ufficiale `calibrated-similarity`
    if HAS_CALIBRATED_SIM:
        try:
            sig = inspect.signature(calibrate_layers)
            params = list(sig.parameters.keys())
            kwargs = {}
            for p_name in params:
                if p_name in ("alpha", "sig_level"):
                    kwargs[p_name] = alpha
                elif p_name in ("n_permutations", "num_permutations", "K", "n_perms"):
                    kwargs[p_name] = n_perms
                elif p_name in ("agg", "aggregation"):
                    kwargs[p_name] = "max"

            # Verifica ordine parametri posizionali (sim_fn, layers_a, layers_b) o (layers_a, layers_b, sim_fn)
            if params and params[0] in ("sim_fn", "metric", "fn", "similarity_fn", "similarity"):
                res = calibrate_layers(metric_fn, feats_a, feats_b, **kwargs)
            else:
                res = calibrate_layers(feats_a, feats_b, metric_fn, **kwargs)

            if isinstance(res, dict) and "calibrated_matrix" in res:
                cal_mat = np.array(res["calibrated_matrix"], dtype=np.float64)
                tau = float(res.get("tau", 0.0))
                return raw_mat, cal_mat, tau, float(cal_mat.max())
            elif isinstance(res, (tuple, list)) and len(res) >= 2:
                # Se restituisce (cal_score, ...) o matrice
                first = np.array(res[0], dtype=np.float64)
                if first.shape == (L, L):
                    return raw_mat, first, 0.0, float(first.max())
        except Exception:
            pass  # Fallback trasparente all'implementazione vettorizzata sotto

    # Implementazione esatta Algorithm 1 & 2 (Gröger et al., 2026):
    # Per ogni permutazione pi_k condivisa sui 49 pair, calcoliamo il massimo sulla griglia 7x7
    gen = torch.Generator(device=feats_a[0].device)
    gen.manual_seed(seed)

    null_max_scores = []
    for _ in range(n_perms):
        perm = torch.randperm(N, generator=gen, device=feats_a[0].device)
        max_k = -1.0
        for i in range(L):
            Xi = feats_a[i]
            for j in range(L):
                Yj = feats_b[j][perm]
                val = float(metric_fn(Xi, Yj).item())
                if val > max_k:
                    max_k = val
        null_max_scores.append(max_k)

    # Statistica d'ordine ceil((1 - alpha) * (K + 1)) su {obs_max} U null_max_scores
    obs_max = float(raw_mat.max())
    combined = sorted([obs_max] + null_max_scores)
    rank_idx = min(len(combined) - 1, math.ceil((1.0 - alpha) * (n_perms + 1)) - 1)
    tau_agg = combined[rank_idx]

    # Calibrazione aggregation-aware di ogni cella rispetto a tau_agg (clipped a 0)
    denom = max(1.0 - tau_agg, 1e-8)
    cal_mat = np.clip((raw_mat - tau_agg) / denom, 0.0, 1.0)
    return raw_mat, cal_mat, float(tau_agg), float(cal_mat.max())


# =====================================================================
# 3. ESTRAZIONE FEATURE (FRASI DEDUPLICATE + ISTANZE INTERE)
# =====================================================================
def select_unique_sentence_plan(
    records: List[dict], tokenizer, dot_id: int, max_unique_sents: int = 800
) -> Tuple[List[dict], List[Tuple[int, int]], List[str]]:
    """
    Seleziona indici (record_idx, sent_idx) garantendo frasi testualmente uniche al 100%
    (evita saturazione dei k-vicini sui template ripetitivi di ProofWriter).
    """
    seen_texts = set()
    chosen_records = []
    targets = []  # (local_rec_idx, sent_idx)
    kept_strings = []

    for r in records:
        q_sents = [s.strip() for s in r["question"].split(".") if s.strip()]
        t_sents = [s.strip() for s in r["theory"].split(".") if s.strip()]
        all_sents = q_sents + t_sents

        enc = tokenizer(
            r["question"], r["theory"], truncation=True, max_length=384, return_tensors="pt"
        )
        sids = compute_sentence_ids(
            enc["input_ids"],
            enc["attention_mask"],
            dot_id,
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
        )[0]
        max_sid = int(sids.max().item()) + 1
        if max_sid != len(all_sents):
            continue

        added_here = False
        local_idx = len(chosen_records)
        for j, s_txt in enumerate(all_sents):
            key = s_txt.lower()
            if key not in seen_texts:
                seen_texts.add(key)
                targets.append((local_idx, j))
                kept_strings.append(key)
                added_here = True
                if len(targets) >= max_unique_sents:
                    break
        if added_here:
            chosen_records.append(r)
        if len(targets) >= max_unique_sents:
            break

    return chosen_records, targets, kept_strings


@torch.no_grad()
def extract_both_granularities(
    model,
    sent_records: List[dict],
    sent_targets: List[Tuple[int, int]],
    inst_records: List[dict],
    tokenizer,
    dot_id: int,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Restituisce (unique_sentence_feats_L0..6, full_instance_mean_feats_L0..6) su device."""
    model.eval()
    n_layers = 7

    # 1. Estrai frasi uniche deduplicate
    target_by_rec: Dict[int, List[int]] = {}
    for rec_i, s_j in sent_targets:
        target_by_rec.setdefault(rec_i, []).append(s_j)

    sent_feats = [[] for _ in range(n_layers)]
    for i in range(0, len(sent_records), batch_size):
        batch = sent_records[i : i + batch_size]
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
        sids = compute_sentence_ids(
            ids, mask, dot_id, tokenizer.cls_token_id, tokenizer.sep_token_id
        )

        with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
            out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)

        for l in range(n_layers):
            reprs, _ = pool_sentences(out.hidden_states[l], sids)
            for b_offset in range(len(batch)):
                rec_global = i + b_offset
                for sj in target_by_rec[rec_global]:
                    sent_feats[l].append(reprs[b_offset, sj].float())

    sent_feats_t = [torch.stack(chunks, dim=0) for chunks in sent_feats]

    # 2. Estrai mean-pooling dell'intera istanza (question + theory) su tutte le coppie uniche
    inst_feats = [[] for _ in range(n_layers)]
    for i in range(0, len(inst_records), batch_size):
        batch = inst_records[i : i + batch_size]
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
        m_float = mask.unsqueeze(-1).float()
        denom = m_float.sum(dim=1).clamp(min=1.0)

        with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
            out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)

        for l in range(n_layers):
            mean_vec = (out.hidden_states[l].float() * m_float).sum(dim=1) / denom
            inst_feats[l].append(mean_vec)

    inst_feats_t = [torch.cat(chunks, dim=0) for chunks in inst_feats]
    return sent_feats_t, inst_feats_t


def plot_calibrated_comparison(
    matrices: Dict[str, np.ndarray], title_main: str, out_path: str
):
    items = list(matrices.items())
    n = len(items)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(
        rows, 2, figsize=(10.2, 4.4 * rows), constrained_layout=True
    )
    axes_flat = np.array(axes).reshape(-1)

    for idx, (title, mat) in enumerate(items):
        ax = axes_flat[idx]
        im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis", origin="lower")
        ax.set_title(title, fontsize=10.5, pad=6)
        ax.set_xlabel("Model B Layer (0..6)", fontsize=9.5)
        ax.set_ylabel("Model A Layer (0..6)", fontsize=9.5)
        ax.set_xticks(range(7))
        ax.set_yticks(range(7))
        for i in range(7):
            for j in range(7):
                val = mat[i, j]
                color = "black" if val > 0.6 else "white"
                ax.text(
                    j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7.5
                )

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(title_main, fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=axes_flat[:n].tolist(), shrink=0.8, aspect=25)
    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="distilroberta-base")
    parser.add_argument("--n_unique_sents", type=int, default=800)
    parser.add_argument("--n_inst", type=int, default=800)
    parser.add_argument("--n_perms", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"calibrated-similarity importato: {HAS_CALIBRATED_SIM} | device={device}")
    if HAS_CALIBRATED_SIM:
        print("  calibrate signature:", inspect.signature(calibrate))
        print("  calibrate_layers signature:", inspect.signature(calibrate_layers))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dot_id = tokenizer.encode(".", add_special_tokens=False)[0]
    _, _, _, test_id_ds, _ = load_or_build_datasets("data_cache", data_seed=42)

    print(f"Selezione di {args.n_unique_sents} frasi uniche deduplicate...")
    sent_recs, sent_targets, _ = select_unique_sentence_plan(
        test_id_ds.records, tokenizer, dot_id, max_unique_sents=args.n_unique_sents
    )
    inst_recs = test_id_ds.records[: args.n_inst]
    print(f"  Frasi uniche estratte: {len(sent_targets)} (da {len(sent_recs)} record) | Istanze uniche: {len(inst_recs)}")

    paths = {
        "free_s42": os.path.join(args.save_dir, "free_seed42", "best_model.pt"),
        "free_s123": os.path.join(args.save_dir, "free_seed123", "best_model.pt"),
        "free_aug_s42": os.path.join(args.save_dir, "free_aug_seed42", "best_model.pt"),
        "const05_s42": os.path.join(args.save_dir, "constrained_seed42_l3_lam0.5", "best_model.pt"),
        "const05_s123": os.path.join(args.save_dir, "constrained_seed123_l3_lam0.5", "best_model.pt"),
        "const10_s42": os.path.join(args.save_dir, "constrained_seed42_l3_lam1.0", "best_model.pt"),
    }

    sent_feats_all = {}
    inst_feats_all = {}
    for name, p in paths.items():
        if not os.path.exists(p):
            continue
        print(f"Estrazione feature per '{name}'...")
        m = load_hf_model(args.model_name, p, device)
        sf, inf = extract_both_granularities(
            m, sent_recs, sent_targets, inst_recs, tokenizer, dot_id, device
        )
        sent_feats_all[name] = sf
        inst_feats_all[name] = inf

    pair_specs = [
        ("free (s42) vs. free (s123)", "free_s42", "free_s123"),
        ("const (λ=0.5, s42) vs. const (λ=0.5, s123)", "const05_s42", "const05_s123"),
        ("free (s42) vs. const (λ=0.5, s42)", "free_s42", "const05_s42"),
        ("free (s42) vs. const (λ=1.0, s42)", "free_s42", "const10_s42"),
        ("free (s42) vs. free_aug (s42)", "free_s42", "free_aug_s42"),
    ]

    metrics = {
        "cka_linear": cka_linear,
        "mknn_k10": make_mknn_fn(10),
        "mknn_k25": make_mknn_fn(25),
        "cknna_k10": make_cknna_fn(10),
    }

    report = {"unique_sentences": {}, "full_instances": {}}
    plot_mats_sent_cka = {}
    plot_mats_sent_mknn10 = {}
    plot_mats_inst_mknn10 = {}

    for gran_name, feats_dict in [
        ("unique_sentences", sent_feats_all),
        ("full_instances", inst_feats_all),
    ]:
        print(f"\n=== CALIBRAZIONE NULL ({gran_name}, K={args.n_perms}, alpha={args.alpha}) ===")
        for title, a, b in pair_specs:
            if a not in feats_dict or b not in feats_dict:
                continue
            pair_key = f"{a}_vs_{b}"
            report[gran_name][pair_key] = {}
            print(f"\nCoppia: {title}")
            for m_name, m_fn in metrics.items():
                raw_m, cal_m, tau, max_cal = calibrate_layer_matrix(
                    feats_dict[a],
                    feats_dict[b],
                    m_fn,
                    n_perms=args.n_perms,
                    alpha=args.alpha,
                )
                report[gran_name][pair_key][m_name] = {
                    "tau_agg": tau,
                    "raw_diag": np.diag(raw_m).round(4).tolist(),
                    "cal_diag": np.diag(cal_m).round(4).tolist(),
                    "raw_full": raw_m.round(4).tolist(),
                    "cal_full": cal_m.round(4).tolist(),
                }
                print(
                    f"  {m_name:10s} | tau={tau:.3f} | "
                    f"raw_L3={raw_m[3,3]:.3f} -> cal_L3={cal_m[3,3]:.3f} | "
                    f"cal_L6={cal_m[6,6]:.3f}"
                )
                if gran_name == "unique_sentences" and title != "free (s42) vs. free_aug (s42)":
                    if m_name == "cka_linear":
                        plot_mats_sent_cka[title] = cal_m
                    elif m_name == "mknn_k10":
                        plot_mats_sent_mknn10[title] = cal_m
                if gran_name == "full_instances" and title != "free (s42) vs. free_aug (s42)":
                    if m_name == "mknn_k10":
                        plot_mats_inst_mknn10[title] = cal_m

    out_dir = os.path.join(args.save_dir, "analysis_aristotelian")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "aristotelian_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    plot_calibrated_comparison(
        plot_mats_sent_cka,
        "Null-Calibrated Linear CKA (Unique Sentences, K=200, α=0.05)",
        os.path.join(out_dir, "cal_cka_sentences_7x7.png"),
    )
    plot_calibrated_comparison(
        plot_mats_sent_mknn10,
        "Null-Calibrated mKNN (k=10, Centered Unique Sentences, K=200)",
        os.path.join(out_dir, "cal_mknn10_sentences_7x7.png"),
    )
    plot_calibrated_comparison(
        plot_mats_inst_mknn10,
        "Null-Calibrated mKNN (k=10, Centered Full Instances, K=200)",
        os.path.join(out_dir, "cal_mknn10_instances_7x7.png"),
    )
    print(f"\nSalvato tutto in: {out_dir}")


if __name__ == "__main__":
    main()