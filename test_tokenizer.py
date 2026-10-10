import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_tokenizer import SimpleTokenizer

TOK_PATH = Path(__file__).resolve().parents[1] / "tokenizer.json"


def test_roundtrip():
    assert TOK_PATH.exists(), f"tokenizer not found: {TOK_PATH}"
    tok = SimpleTokenizer.from_file(TOK_PATH)
    cases = [
        "hello world",
        "Once upon a time",
        "The quick brown fox jumps over the lazy dog",
        "hello, world!",
        "hello  world",
        " leading and trailing ",
        "don't stop",
        "price $100.00 (50% off!)",
        "a\tb\nc",
        "café — test",
        "résumé",
        "foo-bar: a:b;c",
        "   multiple   spaces   ",
        "line1\nline2\nline3",
        "Hello,  world!!  test...",
    ]
    for text in cases:
        ids, back, ok = tok.roundtrip(text)
        print(f"{text!r} -> {len(ids)} ids ok={ok}")
        assert ok, f"roundtrip mismatch: {text!r} -> {back!r}"
        assert len(ids) > 0
        assert all(0 <= i < len(tok.vocab) for i in ids)
