import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from model import build_model


def test_loss_drops_over_100_steps():
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float32)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    m = build_model("smoke", seed=42)
    m.train()
    m.to(torch.float32)
    vs = int(m.cfg.vocab_size)

    g = torch.Generator()
    g.manual_seed(42)
    B, T = 4, 32
    data = torch.randint(0, vs, (B, T), generator=g)

    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    with torch.no_grad():
        init_loss = float(F.cross_entropy(m(data).view(-1, vs), data.view(-1)))
    print(f"init loss={init_loss:.4f}")

    for _ in range(100):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m(data).view(-1, vs), data.view(-1))
        loss.backward()
        opt.step()

    with torch.no_grad():
        final_loss = float(F.cross_entropy(m(data).view(-1, vs), data.view(-1)))
    print(f"final loss={final_loss:.4f} after 100 steps")
    assert final_loss < init_loss, f"loss did not drop: {init_loss:.4f} -> {final_loss:.4f}"
