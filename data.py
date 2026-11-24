from collections import defaultdict
import random
import re
from typing import Dict, List, Tuple

from datasets import load_dataset
import torch
from torch.utils.data import Dataset

# =====================================================================
# 1. VOCABOLARIO ESATTO DI PROOFWRITER (escluso NatLang)
# =====================================================================
SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]

PEOPLE = ["anne", "bob", "charlie", "dave", "erin", "fiona", "gary", "harry"]
ANIMALS = [
    "bald_eagle",
    "bear",
    "cat",
    "cow",
    "dog",
    "lion",
    "mouse",
    "rabbit",
    "squirrel",
    "tiger",
]
ATTRIBUTES = [
    "big",
    "blue",
    "cold",
    "furry",
    "green",
    "kind",
    "nice",
    "quiet",
    "red",
    "rough",
    "round",
    "smart",
    "white",
    "young",
]
VERB_PAIRS = [
    ("chase", "chases"),
    ("eat", "eats"),
    ("like", "likes"),
    ("need", "needs"),
    ("see", "sees"),
    ("visit", "visits"),
]
SYNTAX_WORDS = [
    ",",
    ".",
    "all",
    "and",
    "are",
    "do",
    "does",
    "if",
    "is",
    "it",
    "not",
    "people",
    "someone",
    "something",
    "the",
    "then",
    "they",
    "things",
]

LABEL_MAP = {"False": 0, "True": 1, "Unknown": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


class ProofWriterTokenizer:

    def __init__(self):
        verbs_flat = [w for pair in VERB_PAIRS for w in pair]
        self.vocab = (
            SPECIAL_TOKENS
            + SYNTAX_WORDS
            + PEOPLE
            + ANIMALS
            + ATTRIBUTES
            + verbs_flat
        )
        self.token_to_id: Dict[str, int] = {
            tok: i for i, tok in enumerate(self.vocab)
        }
        self.id_to_token: Dict[int, str] = {
            i: tok for i, tok in enumerate(self.vocab)
        }

        self.pad_id = self.token_to_id["[PAD]"]
        self.cls_id = self.token_to_id["[CLS]"]
        self.sep_id = self.token_to_id["[SEP]"]
        self.unk_id = self.token_to_id["[UNK]"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @staticmethod
    def tokenize_text(text: str) -> List[str]:
        text = text.lower()
        # Unifica "bald eagle" in un singolo token atomico
        text = re.sub(r"\bbald eagle\b", "bald_eagle", text)
        return re.findall(r"[a-z0-9_']+|[.,]", text)

    def encode(
        self, theory: str, question: str, max_len: int = 320
    ) -> Tuple[List[int], List[int], bool]:
        t_tokens = self.tokenize_text(theory)
        q_tokens = self.tokenize_text(question)

        # Domanda fissa davanti, subito dopo [CLS], per facilitare la ricerca nella teoria
        tokens = ["[CLS]"] + q_tokens + ["[SEP]"] + t_tokens + ["[SEP]"]
        truncated = len(tokens) > max_len
        tokens = tokens[:max_len]

        ids = [self.token_to_id.get(tok, self.unk_id) for tok in tokens]
        attn_mask = [1] * len(ids)

        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids.extend([self.pad_id] * pad_len)
            attn_mask.extend([0] * pad_len)

        return ids, attn_mask, truncated

    def decode(self, ids: List[int]) -> str:
        toks = [
            self.id_to_token[i]
            for i in ids
            if i != self.pad_id and i in self.id_to_token
        ]
        return " ".join(toks)


class AlphaPermutator:
    """Applica una permutazione simbolica biettiva (alpha-equivalenza) sui token ID."""

    def __init__(self, tokenizer: ProofWriterTokenizer):
        self.tok = tokenizer
        self.vocab_size = tokenizer.vocab_size

        self.people_ids = [tokenizer.token_to_id[w] for w in PEOPLE]
        self.animal_ids = [tokenizer.token_to_id[w] for w in ANIMALS]
        self.attr_ids = [tokenizer.token_to_id[w] for w in ATTRIBUTES]
        self.verb_pair_ids = [
            (tokenizer.token_to_id[b], tokenizer.token_to_id[s])
            for b, s in VERB_PAIRS
        ]

    def sample_perm_table(self, rng: random.Random) -> torch.Tensor:
        table = list(range(self.vocab_size))

        # Permuta persone <-> persone
        p_shuf = self.people_ids.copy()
        rng.shuffle(p_shuf)
        for src, dst in zip(self.people_ids, p_shuf):
            table[src] = dst

        # Permuta animali <-> animali
        a_shuf = self.animal_ids.copy()
        rng.shuffle(a_shuf)
        for src, dst in zip(self.animal_ids, a_shuf):
            table[src] = dst

        # Permuta attributi <-> attributi
        attr_shuf = self.attr_ids.copy()
        rng.shuffle(attr_shuf)
        for src, dst in zip(self.attr_ids, attr_shuf):
            table[src] = dst

        # Permuta verbi in coppia (base <-> base, -s <-> -s)
        v_shuf = self.verb_pair_ids.copy()
        rng.shuffle(v_shuf)
        for (b_src, s_src), (b_dst, s_dst) in zip(self.verb_pair_ids, v_shuf):
            table[b_src] = b_dst
            table[s_src] = s_dst

        return torch.tensor(table, dtype=torch.long)

    def permute_batch(
        self, input_ids: torch.Tensor, rng: random.Random
    ) -> torch.Tensor:
        """input_ids: (B, T) -> restituisce (B, T) con una permutazione indipendente per ogni esempio."""
        tables = torch.stack(
            [self.sample_perm_table(rng) for _ in range(input_ids.size(0))],
            dim=0,
        ).to(input_ids.device)
        # Per ogni riga b, applica tables[b, input_ids[b]]
        return torch.gather(tables, 1, input_ids)


class ProofWriterDataset(Dataset):

    def __init__(
        self,
        records: List[dict],
        tokenizer: ProofWriterTokenizer,
        max_len: int = 320,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.input_ids = []
        self.attention_mask = []
        self.labels = []
        self.qdeps = []
        self.configs = []

        n_truncated = 0
        n_unk = 0
        for r in records:
            ids, mask, trunc = tokenizer.encode(
                r["theory"], r["question"], max_len=max_len
            )
            if trunc:
                n_truncated += 1
            if tokenizer.unk_id in ids:
                n_unk += 1
            self.input_ids.append(ids)
            self.attention_mask.append(mask)
            self.labels.append(LABEL_MAP[r["answer"]])
            self.qdeps.append(int(r["QDep"]))
            self.configs.append(r["config"])

        self.input_ids = torch.tensor(self.input_ids, dtype=torch.long)
        self.attention_mask = torch.tensor(
            self.attention_mask, dtype=torch.long
        )
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.qdeps = torch.tensor(self.qdeps, dtype=torch.long)
        self.n_truncated = n_truncated
        self.n_unk = n_unk

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx],
            "qdep": self.qdeps[idx],
        }


def _sample_balanced_id(
    rows: List[dict],
    total_samples: int,
    seed: int,
) -> List[dict]:
    """Campiona in modo bilanciato tra (True, False, Unknown) e per QDep in {0, 1, 2}."""
    rng = random.Random(seed)
    per_class = total_samples // 3

    by_class_qdep = defaultdict(list)
    unknown_rows = []
    for r in rows:
        if r["answer"] in ("True", "False"):
            by_class_qdep[(r["answer"], int(r["QDep"]))].append(r)
        else:
            unknown_rows.append(r)

    selected = []
    for cls_name in ("True", "False"):
        per_qdep = per_class // 3
        for qd in (0, 1, 2):
            pool = by_class_qdep[(cls_name, qd)]
            rng.shuffle(pool)
            selected.extend(pool[:per_qdep])

    # Per Unknown bilanciamo sui config (depth-0, depth-1, depth-2)
    unk_by_cfg = defaultdict(list)
    for r in unknown_rows:
        unk_by_cfg[r["config"]].append(r)
    per_cfg = per_class // 3
    for cfg in ("depth-0", "depth-1", "depth-2"):
        pool = unk_by_cfg[cfg]
        rng.shuffle(pool)
        selected.extend(pool[:per_cfg])

    rng.shuffle(selected)
    return selected


def build_datasets(
    max_len: int = 320,
    train_size: int = 45000,
    val_size: int = 6000,
    test_id_size: int = 6000,
    test_ood_size: int = 12000,
    seed: int = 42,
):
    print("Caricamento tasksource/proofwriter...")
    raw = load_dataset("tasksource/proofwriter")
    tokenizer = ProofWriterTokenizer()

    id_configs = {"depth-0", "depth-1", "depth-2"}
    ood_configs = {"depth-3", "depth-5", "depth-3ext"}

    # 1. Estrai train ID (QDep <= 2)
    train_pool = [
        r
        for r in raw["train"]
        if r["config"] in id_configs and int(r["QDep"]) <= 2
    ]
    train_records = _sample_balanced_id(train_pool, train_size, seed=seed)
    train_theories = set(r["theory"] for r in train_pool)

    # 2. Validation ID (verifica 0 theory overlap con train)
    val_pool = [
        r
        for r in raw["validation"]
        if r["config"] in id_configs
        and int(r["QDep"]) <= 2
        and r["theory"] not in train_theories
    ]
    val_records = _sample_balanced_id(val_pool, val_size, seed=seed + 1)

    # 3. Test ID (verifica 0 theory overlap con train)
    test_id_pool = [
        r
        for r in raw["test"]
        if r["config"] in id_configs
        and int(r["QDep"]) <= 2
        and r["theory"] not in train_theories
    ]
    test_id_records = _sample_balanced_id(
        test_id_pool, test_id_size, seed=seed + 2
    )

    # 4. Test OOD (depth-3, depth-5, depth-3ext; bilanciato su QDep per True/False + Unknown)
    rng = random.Random(seed + 3)
    ood_pool = [
        r
        for r in raw["test"]
        if r["config"] in ood_configs and r["theory"] not in train_theories
    ]
    # Campioniamo includendo bene tutte le profondità 0..5
    ood_by_key = defaultdict(list)
    for r in ood_pool:
        qd = min(int(r["QDep"]), 5)
        ood_by_key[(r["answer"], qd)].append(r)

    test_ood_records = []
    # Prendiamo fino a ~500 esempi per ogni coppia (answer, qdep 0..5) dove disponibili
    target_per_bucket = max(1, test_ood_size // 18)
    for key, pool in sorted(ood_by_key.items()):
        rng.shuffle(pool)
        test_ood_records.extend(pool[:target_per_bucket])
    rng.shuffle(test_ood_records)

    print("Tokenizzazione dataset...")
    train_ds = ProofWriterDataset(train_records, tokenizer, max_len=max_len)
    val_ds = ProofWriterDataset(val_records, tokenizer, max_len=max_len)
    test_id_ds = ProofWriterDataset(test_id_records, tokenizer, max_len=max_len)
    test_ood_ds = ProofWriterDataset(
        test_ood_records, tokenizer, max_len=max_len
    )

    return tokenizer, train_ds, val_ds, test_id_ds, test_ood_ds


if __name__ == "__main__":
    tok, train_ds, val_ds, test_id_ds, test_ood_ds = build_datasets()
    print(f"\nVocab size: {tok.vocab_size}")
    for name, ds in [
        ("Train (ID)", train_ds),
        ("Val (ID)", val_ds),
        ("Test (ID)", test_id_ds),
        ("Test (OOD)", test_ood_ds),
    ]:
        print(
            f"{name:12s} | N={len(ds):5d} | truncated={ds.n_truncated} | unk={ds.n_unk}"
        )

    # Test visivo del permutatore alpha-equivalente
    perm = AlphaPermutator(tok)
    rng = random.Random(0)
    sample_x = train_ds[0]["input_ids"].unsqueeze(0)
    sample_px = perm.permute_batch(sample_x, rng)

    print("\n=== TEST PERMUTAZIONE ALPHA-EQUIVALENTE ===")
    print("Originale: ", tok.decode(sample_x[0].tolist())[:220], "...")
    print("Permutato: ", tok.decode(sample_px[0].tolist())[:220], "...")