from collections import Counter, defaultdict
import re
from datasets import load_dataset


def simple_words(text: str):
    return re.findall(r"[A-Za-z0-9_']+|[.,!?;]", text.lower())


def main():
    ds = load_dataset("tasksource/proofwriter", split="train")

    vocab = Counter()
    seen_theories = set()
    max_len_by_cfg = defaultdict(int)
    qdep_by_ans_d5 = defaultdict(Counter)
    id_prefixes = Counter()

    for row in ds:
        cfg = row["config"]
        if "NatLang" in cfg:
            continue

        # Prefisso ID (es. AttNeg, RelNoneg, ecc.)
        id_prefixes[row["id"].split("-")[0]] += 1

        # Lunghezza approssimata veloce in parole
        approx_len = len(row["theory"].split()) + len(row["question"].split()) + 10
        if approx_len > max_len_by_cfg[cfg]:
            max_len_by_cfg[cfg] = approx_len

        if cfg == "depth-5":
            qdep_by_ans_d5[row["answer"]][row["QDep"]] += 1

        # Tokenizza ogni teoria unica una sola volta (velocissimo!)
        th = row["theory"]
        if th not in seen_theories:
            seen_theories.add(th)
            vocab.update(simple_words(th))
        vocab.update(simple_words(row["question"]))

    print("=== PREFISSI ID (tipi di teorie) ===")
    print(dict(id_prefixes))

    print("\n=== LUNGHEZZA MASSIMA STIMATA PER CONFIG ===")
    print(dict(sorted(max_len_by_cfg.items())))

    print("\n=== QDep per Answer su depth-5 ===")
    for ans, cnt in qdep_by_ans_d5.items():
        print(f"  {ans}: {dict(sorted(cnt.items()))}")

    print(f"\n=== VOCABOLARIO COMPLETO ({len(vocab)} parole uniche) ===")
    all_words = sorted(vocab.keys())
    print(all_words)


if __name__ == "__main__":
    main()