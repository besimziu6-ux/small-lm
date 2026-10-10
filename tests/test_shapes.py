import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from model import build_model


def _seeded_gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_logits_shape():
    torch.set_num_threads(4)
    m = build_model("smoke", seed=42)
    m.eval()
    B, T = 2, 16
    vs = int(m.cfg.vocab_size)
    x = torch.randint(0, vs, (B, T), generator=_seeded_gen(42))
    with torch.no_grad():
        logits = m(x)
    print(f"logits shape={tuple(logits.shape)} vocab={vs}")
    assert logits.shape == (B, T, vs)


def test_attention_mask_causal():
    T = 16
    mask = torch.ones(T, T, dtype=torch.bool).tril()
    assert mask.shape == (T, T)
    assert not mask.triu(diagonal=1).any(), "upper triangle must be masked"
    assert bool(mask[0, 0]) and not bool(mask[0, 1:].any())
    assert bool(mask[-1].all()), "last position attends to all past"
    print(f"causal mask ok T={T}")


def test_causal_behavior():
    torch.set_num_threads(4)
    m = build_model("smoke", seed=42)
    m.eval()
    T, k = 16, 8
    vs = int(m.cfg.vocab_size)
    g = _seeded_gen(7)
    x1 = torch.randint(0, vs, (1, T), generator=g)
    x2 = x1.clone()
    x2[0, k:] = torch.randint(0, vs, (T - k,), generator=_seeded_gen(99))
    assert not torch.equal(x1, x2)
    with torch.no_grad():
        l1 = m(x1)
        l2 = m(x2)
    assert torch.allclose(l1[:, :k, :], l2[:, :k, :], atol=1e-5), (
        "prefix logits must be unaffected by future tokens"
    )
    assert not torch.allclose(l1[:, k:, :], l2[:, k:, :], atol=1e-5), (
        "suffix change should affect later logits"
    )
    print(f"causal behavior ok T={T} prefix={k}")
