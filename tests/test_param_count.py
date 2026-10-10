import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import build_model
from utils.params import count_parameters


def test_base_preset_in_50_to_100M():
    m = build_model("base", seed=42)
    total, _ = count_parameters(m)
    print(f"base total={total:,}")
    assert 50_000_000 <= total <= 100_000_000, f"base {total:,} outside 50M-100M"


def test_smoke_under_10M():
    m = build_model("smoke", seed=42)
    total, _ = count_parameters(m)
    print(f"smoke total={total:,}")
    assert total < 10_000_000, f"smoke {total:,} not under 10M"
