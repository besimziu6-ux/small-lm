import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from model import build_model


def test_same_seed_same_init():
    torch.set_num_threads(4)
    m1 = build_model("smoke", seed=123)
    m2 = build_model("smoke", seed=123)
    p1 = dict(m1.named_parameters())
    p2 = dict(m2.named_parameters())
    assert set(p1) == set(p2)
    for name in p1:
        assert torch.equal(p1[name].cpu(), p2[name].cpu()), f"param differs: {name}"
    print(f"seed=123 init identical params={len(p1)}")


def test_same_seed_same_forward():
    torch.set_num_threads(4)
    m1 = build_model("smoke", seed=123)
    m2 = build_model("smoke", seed=123)
    m1.eval()
    m2.eval()
    vs = int(m1.cfg.vocab_size)
    g = torch.Generator()
    g.manual_seed(123)
    x = torch.randint(0, vs, (2, 16), generator=g)
    with torch.no_grad():
        l1 = m1(x)
        l2 = m2(x)
    assert torch.equal(l1, l2), "same seed must give identical forward"
    print(f"seed=123 forward identical shape={tuple(l1.shape)}")
