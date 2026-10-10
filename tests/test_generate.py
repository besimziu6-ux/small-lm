import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from generate import sample_next_id
from model import build_model


def _greedy_ids(seed: int, max_new: int) -> list[int]:
    torch.set_num_threads(4)
    m = build_model("smoke", seed=seed)
    m.eval()
    vs = int(m.cfg.vocab_size)
    ctx = [2, 100 % vs, 200 % vs, 300 % vs]
    new_ids: list[int] = []
    with torch.no_grad():
        for _ in range(max_new):
            window = ctx[-int(m.cfg.max_seq_len):]
            x = torch.tensor([window], dtype=torch.long)
            logits = m(x)
            nid = sample_next_id(logits[0, -1, :], 0.0, 0, 1.0)
            new_ids.append(int(nid))
            ctx.append(int(nid) if 0 <= int(nid) < vs else 1)
    return new_ids


def test_greedy_output_length_and_vocab():
    m = build_model("smoke", seed=42)
    vs = int(m.cfg.vocab_size)
    max_new = 20
    new_ids = _greedy_ids(42, max_new)
    print(f"greedy ids={new_ids}")
    assert len(new_ids) == max_new, f"expected {max_new}, got {len(new_ids)}"
    assert all(0 <= i < vs for i in new_ids), "ids must be valid vocab ids"


def test_greedy_deterministic():
    a = _greedy_ids(42, 20)
    b = _greedy_ids(42, 20)
    assert a == b, "greedy decode must be deterministic for same seed"
    print("greedy deterministic ok")
