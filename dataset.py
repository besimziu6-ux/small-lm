from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import Config
from utils.paths import resolve_path
from utils.seed import set_all_seeds


def _safe_seed(seed: int) -> None:
    try:
        set_all_seeds(seed)
    except RuntimeError:
        import random

        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

try:
    from train_tokenizer import (
        BOS_ID,
        EOS_ID,
        PAD_ID,
        UNK_ID,
        SimpleTokenizer,
        load_tokenizer_file,
    )
except ImportError:
    from pathlib import Path as _P
    import json as _json

    PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

    def load_tokenizer_file(path):  # type: ignore[misc]
        data = _json.loads(_P(path).read_text(encoding="utf-8"))
        return data["vocab"], data["merges"], {}

    class SimpleTokenizer:  # type: ignore[no-redef]
        def __init__(self, vocab, merges):
            self.vocab = vocab
            self.merges = merges
            self.pad_id = PAD_ID
            self.unk_id = UNK_ID
            self.bos_id = BOS_ID
            self.eos_id = EOS_ID
            self.inv_vocab = {v: k for k, v in vocab.items()}

        @classmethod
        def from_file(cls, path):
            v, m, _ = load_tokenizer_file(path)
            return cls(v, m)

        def encode(self, text, add_special_tokens=False):
            return []

        def decode(self, ids, skip_special_tokens=True):
            return ""

SEQ_LEN = 256
VAL_SPLIT = 0.05


def _split_docs(text: str) -> list[str]:
    docs = [d.strip() for d in text.split("\n") if d.strip()]
    if len(docs) >= 10:
        return docs
    chunks = [text[i : i + 2000] for i in range(0, len(text), 2000)]
    chunks = [c.strip() for c in chunks if c.strip()]
    return chunks or [text.strip()]


def tokenize_stream(raw_path: str | Path, tokenizer_path: str | Path, max_doc_tokens: int = 1024) -> list[int]:
    rp = resolve_path(raw_path)
    tp = resolve_path(tokenizer_path)
    tok = SimpleTokenizer.from_file(tp)
    text = rp.read_text(encoding="utf-8", errors="ignore")
    docs = _split_docs(text)
    stream: list[int] = []
    for d in docs:
        ids = tok.encode(d)[:max_doc_tokens]
        stream.append(tok.bos_id)
        stream.extend(ids)
        stream.append(tok.eos_id)
    return stream


def split_train_val(ids: Sequence[int], val_split: float = VAL_SPLIT) -> tuple[list[int], list[int]]:
    n = len(ids)
    if n == 0:
        return [], []
    n_val = int(n * val_split)
    n_train = n - n_val
    if n_train <= 0:
        n_train = n
        n_val = 0
    return list(ids[:n_train]), list(ids[n_train:])


def chunk_blocks(ids: Sequence[int], seq_len: int = SEQ_LEN, pad_id: int = PAD_ID) -> list[list[int]]:
    width = seq_len + 1
    blocks: list[list[int]] = []
    for i in range(0, len(ids), width):
        blk = list(ids[i : i + width])
        if len(blk) < width:
            blk = blk + [pad_id] * (width - len(blk))
        blocks.append(blk)
    return blocks


def ensure_min_tokens(ids: list[int], seq_len: int, min_blocks: int, pad_id: int) -> list[int]:
    need = min_blocks * (seq_len + 1)
    if len(ids) >= need:
        return ids
    if not ids:
        return [pad_id] * need
    tiled = (ids * ((need // len(ids)) + 1))[:need]
    return tiled


class CausalLMDataset:
    def __init__(self, token_ids: Sequence[int], seq_len: int = SEQ_LEN, pad_id: int = PAD_ID) -> None:
        import torch

        self.seq_len = seq_len
        self.pad_id = pad_id
        blocks = chunk_blocks(list(token_ids), seq_len=seq_len, pad_id=pad_id)
        inputs = []
        labels = []
        for blk in blocks:
            x = blk[:-1]
            y = blk[1:]
            y = [t if t != pad_id else -100 for t in y]
            inputs.append(x)
            labels.append(y)
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> dict[str, object]:
        return {"input_ids": self.inputs[idx], "labels": self.labels[idx]}


def build_datasets(
    raw_path: str | Path = "data/raw/sample.txt",
    tokenizer_path: str | Path = "tokenizer.json",
    seq_len: int = SEQ_LEN,
    val_split: float = VAL_SPLIT,
    seed: int = 42,
    min_train_blocks: int = 8,
) -> tuple[CausalLMDataset, CausalLMDataset]:
    _safe_seed(seed)
    stream = tokenize_stream(raw_path, tokenizer_path)
    train_ids, val_ids = split_train_val(stream, val_split)
    tok = SimpleTokenizer.from_file(resolve_path(tokenizer_path))
    pad = tok.pad_id
    train_ids = ensure_min_tokens(train_ids, seq_len, min_train_blocks, pad)
    if len(val_ids) < seq_len + 1:
        val_ids = ensure_min_tokens(val_ids or train_ids, seq_len, 1, pad)
    return CausalLMDataset(train_ids, seq_len, pad), CausalLMDataset(val_ids, seq_len, pad)


def get_dataloaders(
    raw_path: str | Path = "data/raw/sample.txt",
    tokenizer_path: str | Path = "tokenizer.json",
    seq_len: int = SEQ_LEN,
    batch_size: int = 8,
    seed: int = 42,
) -> tuple[object, object]:
    import torch

    _safe_seed(seed)
    train_ds, val_ds = build_datasets(raw_path, tokenizer_path, seq_len=seq_len, seed=seed)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, generator=g)  # type: ignore[arg-type]
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def roundtrip(text: str, tokenizer_path: str | Path = "tokenizer.json") -> tuple[list[int], str, bool]:
    tok = SimpleTokenizer.from_file(resolve_path(tokenizer_path))
    ids, back, ok = tok.roundtrip(text)
    return ids, back, ok


def encode(text: str, tokenizer_path: str | Path = "tokenizer.json", add_special_tokens: bool = False) -> list[int]:
    tok = SimpleTokenizer.from_file(resolve_path(tokenizer_path))
    return tok.encode(text, add_special_tokens=add_special_tokens)


def decode(ids: Sequence[int], tokenizer_path: str | Path = "tokenizer.json") -> str:
    tok = SimpleTokenizer.from_file(resolve_path(tokenizer_path))
    return tok.decode(list(ids))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/raw/sample.txt")
    p.add_argument("--tokenizer", default="tokenizer.json")
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    cfg = Config()
    _ = cfg
    _safe_seed(args.seed)
    train_loader, val_loader = get_dataloaders(args.raw, args.tokenizer, seq_len=args.seq_len, batch_size=args.batch_size, seed=args.seed)
    batch = next(iter(train_loader))
    print(f"input_ids={tuple(batch['input_ids'].shape)} labels={tuple(batch['labels'].shape)}")
    print(f"min_id={int(batch['input_ids'].min())} max_id={int(batch['input_ids'].max())}")


if __name__ == "__main__":
    main()
