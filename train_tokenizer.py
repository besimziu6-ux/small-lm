from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import Config
from utils.paths import resolve_path
from utils.seed import set_all_seeds

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
SPECIAL_IDS = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID, BOS_TOKEN: BOS_ID, EOS_TOKEN: EOS_ID}

TOKEN_RE = re.compile(r"\S+|\s+")


def _build_bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_BYTE_TO_UNICODE: dict[int, str] = _build_bytes_to_unicode()
_UNICODE_TO_BYTES: dict[str, int] = {v: k for k, v in _BYTE_TO_UNICODE.items()}

_HF_BYTELEVEL = None


def _get_hf_bytelevel():
    global _HF_BYTELEVEL
    if _HF_BYTELEVEL is not None:
        return _HF_BYTELEVEL
    try:
        from tokenizers.pre_tokenizers import ByteLevel

        _HF_BYTELEVEL = ByteLevel(add_prefix_space=False)
        return _HF_BYTELEVEL
    except Exception:
        return None


def _byte_encode_piece(s: str) -> str:
    return "".join(_BYTE_TO_UNICODE[b] for b in s.encode("utf-8"))


def _byte_decode_string(s: str) -> str:
    buf = bytearray()
    out: list[str] = []
    for ch in s:
        b = _UNICODE_TO_BYTES.get(ch)
        if b is None:
            if buf:
                out.append(bytes(buf).decode("utf-8", errors="replace"))
                buf.clear()
            out.append(ch)
            continue
        buf.append(b)
    if buf:
        out.append(bytes(buf).decode("utf-8", errors="replace"))
    return "".join(out)


def _byte_level_pieces(text: str) -> list[str]:
    if not text:
        return []
    pt = _get_hf_bytelevel()
    if pt is not None:
        try:
            return [t for t, _ in pt.pre_tokenize_str(text)]
        except Exception:
            pass
    try:
        import regex as _re_mod

        found = _re_mod.findall(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            text,
        )
    except ImportError:
        found = re.findall(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""",
            text,
        )
    if not found:
        return [_byte_encode_piece(text)]
    return [_byte_encode_piece(p) for p in found]


def encode_text_bytelevel(
    text: str, vocab: dict[str, int], rank: dict[tuple[str, str], int], unk_id: int = UNK_ID
) -> list[int]:
    ids: list[int] = []
    for piece in _byte_level_pieces(text):
        for tok in _encode_word(piece, rank):
            ids.append(vocab.get(tok, unk_id))
    return ids


def decode_ids_bytelevel(ids: list[int] | object, inv_vocab: dict[int, str], skip_specials: bool = True) -> str:
    try:
        import torch

        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
    except ImportError:
        pass
    seq = list(ids)  # type: ignore[arg-type]
    skip = {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN} if skip_specials else set()
    parts: list[str] = []
    for i in seq:
        tok = inv_vocab.get(int(i), UNK_TOKEN)  # type: ignore[arg-type]
        if tok in skip:
            continue
        if skip_specials and tok.startswith("<extra_"):
            continue
        parts.append(tok)
    return _byte_decode_string("".join(parts))


def _is_bytelevel_file(path: str | Path) -> bool:
    try:
        p = resolve_path(path)
        if not p.exists():
            alt = SCRIPT_DIR / Path(path).name
            if alt.exists():
                p = alt
            else:
                return False
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("pre_tokenizer") == "bytelevel":
            return True
        pt = data.get("pre_tokenizer")
        if isinstance(pt, dict) and pt.get("type") == "ByteLevel":
            return True
        return False
    except Exception:
        return False


def _collect_files(inp: Path) -> list[Path]:
    if inp.is_dir():
        files = sorted(inp.glob("*.txt"))
        if not files:
            files = sorted(inp.rglob("*.txt"))
        return files
    return [inp]


def _read_texts(files: list[Path]) -> list[str]:
    texts = []
    for f in files:
        texts.append(f.read_text(encoding="utf-8", errors="ignore"))
    return texts


def _merge_word(parts: list[str], a: str, b: str) -> list[str]:
    merged = a + b
    out: list[str] = []
    i = 0
    while i < len(parts):
        if i < len(parts) - 1 and parts[i] == a and parts[i + 1] == b:
            out.append(merged)
            i += 2
        else:
            out.append(parts[i])
            i += 1
    return out


def _encode_word(word: str, rank: dict[tuple[str, str], int]) -> list[str]:
    parts = list(word)
    if len(parts) <= 1:
        return parts
    while len(parts) > 1:
        best_idx = -1
        best_rank: int | None = None
        for i in range(len(parts) - 1):
            r = rank.get((parts[i], parts[i + 1]))
            if r is not None and (best_rank is None or r < best_rank):
                best_rank = r
                best_idx = i
                if r == 0:
                    break
        if best_idx == -1:
            break
        merged = parts[best_idx] + parts[best_idx + 1]
        parts = parts[:best_idx] + [merged] + parts[best_idx + 2:]
    return parts


def encode_text(text: str, vocab: dict[str, int], rank: dict[tuple[str, str], int], unk_id: int = UNK_ID) -> list[int]:
    ids: list[int] = []
    for m in TOKEN_RE.finditer(text):
        part = m.group(0)
        if part.isspace():
            for ch in part:
                ids.append(vocab.get(ch, unk_id))
        else:
            for tok in _encode_word(part, rank):
                ids.append(vocab.get(tok, unk_id))
    return ids


def decode_ids(ids: list[int] | object, inv_vocab: dict[int, str], skip_specials: bool = True) -> str:
    try:
        import torch

        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
    except ImportError:
        pass
    seq = list(ids)  # type: ignore[arg-type]
    skip = {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN} if skip_specials else set()
    out: list[str] = []
    for i in seq:
        tok = inv_vocab.get(int(i), UNK_TOKEN)  # type: ignore[arg-type]
        if tok in skip:
            continue
        if skip_specials and tok.startswith("<extra_"):
            continue
        out.append(tok)
    return "".join(out)


def build_rank(merges: list[str]) -> dict[tuple[str, str], int]:
    rank: dict[tuple[str, str], int] = {}
    for i, m in enumerate(merges):
        if isinstance(m, (list, tuple)):
            parts = [str(x) for x in m]
            if len(parts) != 2:
                continue
            a, b = parts
        else:
            a, _, b = str(m).partition(" ")
        if not a or not b:
            continue
        key = (a, b)
        if key not in rank:
            rank[key] = i
    return rank


def pad_vocab_to_size(vocab: dict[str, int], vocab_size: int) -> dict[str, int]:
    n = len(vocab)
    i = 0
    while len(vocab) < vocab_size:
        cand = f"<extra_{i}>"
        i += 1
        if cand in vocab:
            continue
        vocab[cand] = n
        n += 1
    return vocab


def fallback_train(texts: list[str], vocab_size: int, min_frequency: int = 2) -> tuple[dict[str, int], list[str]]:
    full = "\n".join(texts)
    words = re.findall(r"\S+", full)
    counts = Counter(words)
    if min_frequency > 1:
        filtered = {w: c for w, c in counts.items() if c >= min_frequency}
        if not filtered:
            filtered = dict(counts)
    else:
        filtered = dict(counts)
    if len(filtered) > 30000:
        top = sorted(filtered.items(), key=lambda kv: (-kv[1], kv[0]))[:30000]
        filtered = dict(top)
    vocab: dict[str, int] = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
    charset: set[str] = set()
    for w in filtered:
        charset.update(list(w))
    for i in range(32, 127):
        charset.add(chr(i))
    for ch in (" ", "\n", "\t", "\r"):
        charset.add(ch)
    for ch in set(full):
        charset.add(ch)
        if len(charset) > 10000:
            break
    for ch in sorted(charset):
        if ch not in vocab:
            vocab[ch] = len(vocab)
            if len(vocab) >= vocab_size:
                break
    if len(vocab) >= vocab_size:
        trimmed = dict(list(vocab.items())[:vocab_size])
        return trimmed, []
    splits: dict[str, list[str]] = {w: list(w) for w in filtered}
    merges: list[str] = []
    while len(vocab) < vocab_size:
        pair_counts: Counter[tuple[str, str]] = Counter()
        for w, c in filtered.items():
            s = splits[w]
            if len(s) < 2:
                continue
            for j in range(len(s) - 1):
                pair_counts[(s[j], s[j + 1])] += c
        if not pair_counts:
            break
        max_freq = max(pair_counts.values())
        if max_freq < min_frequency:
            break
        cands = [p for p, f in pair_counts.items() if f == max_freq]
        best = sorted(cands)[0]
        new_tok = best[0] + best[1]
        is_new = new_tok not in vocab
        if is_new:
            vocab[new_tok] = len(vocab)
            merges.append(f"{best[0]} {best[1]}")
        for w in list(splits.keys()):
            s = splits[w]
            if len(s) < 2:
                continue
            splits[w] = _merge_word(s, best[0], best[1])
        if len(vocab) >= vocab_size:
            break
    pad_vocab_to_size(vocab, vocab_size)
    return vocab, merges


def hf_train(files: list[Path], vocab_size: int, min_frequency: int) -> tuple[dict[str, int], list[str]]:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers import decoders, pre_tokenizers
    from tokenizers.trainers import BpeTrainer

    tok = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    initial = pre_tokenizers.ByteLevel.alphabet()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=initial,
        show_progress=False,
    )
    tok.train([str(f) for f in files], trainer)
    vocab = tok.get_vocab()
    raw = json.loads(tok.to_str())
    raw_merges = raw.get("model", {}).get("merges", [])
    merges: list[str] = []
    for m in raw_merges:
        if isinstance(m, (list, tuple)):
            parts = [str(x) for x in m]
            if len(parts) == 2:
                merges.append(f"{parts[0]} {parts[1]}")
        else:
            merges.append(str(m))
    return dict(vocab), merges


def save_tokenizer(path: Path, vocab: dict[str, int], merges: list[str], backend: str, pre_tokenizer: str | None = None) -> None:
    if pre_tokenizer is None:
        pre_tokenizer = "bytelevel" if backend == "hf-bpe" else "whitespace"
    payload = {
        "version": 1,
        "backend": backend,
        "pre_tokenizer": pre_tokenizer,
        "vocab_size": len(vocab),
        "vocab": vocab,
        "merges": merges,
        "special_tokens": {
            "pad": PAD_TOKEN,
            "unk": UNK_TOKEN,
            "bos": BOS_TOKEN,
            "eos": EOS_TOKEN,
        },
        "special_ids": SPECIAL_IDS,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_tokenizer_file(path: str | Path) -> tuple[dict[str, int], list[str], dict[str, int]]:
    p = resolve_path(path)
    if not p.exists():
        alt = SCRIPT_DIR / Path(path).name
        if alt.exists():
            p = alt
    data = json.loads(p.read_text(encoding="utf-8"))
    if "vocab" in data and "merges" in data:
        vocab = {str(k): int(v) for k, v in data["vocab"].items()}
        merges = [f"{m[0]} {m[1]}" if isinstance(m, (list, tuple)) else str(m) for m in data["merges"]]
        specials = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID, BOS_TOKEN: BOS_ID, EOS_TOKEN: EOS_ID}
        if isinstance(data.get("special_ids"), dict):
            try:
                specials = {str(k): int(v) for k, v in data["special_ids"].items()}
            except (ValueError, TypeError) as e:
                print(f"special_ids parse failed ({p}: {e}), using defaults", file=sys.stderr)
        return vocab, merges, specials
    if "model" in data and data["model"].get("type") == "BPE":
        vocab = {str(k): int(v) for k, v in data["model"]["vocab"].items()}
        merges = [str(m) for m in data["model"].get("merges", [])]
        return vocab, merges, SPECIAL_IDS
    raise ValueError(f"unrecognized tokenizer format: {p}")


class SimpleTokenizer:
    def __init__(self, vocab: dict[str, int], merges: list[str], is_bytelevel: bool = False) -> None:
        self.vocab = vocab
        self.merges = merges
        self.rank = build_rank(merges)
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.pad_id = vocab.get(PAD_TOKEN, PAD_ID)
        self.unk_id = vocab.get(UNK_TOKEN, UNK_ID)
        self.bos_id = vocab.get(BOS_TOKEN, BOS_ID)
        self.eos_id = vocab.get(EOS_TOKEN, EOS_ID)
        self.is_bytelevel = is_bytelevel

    @classmethod
    def from_file(cls, path: str | Path) -> SimpleTokenizer:
        vocab, merges, _ = load_tokenizer_file(path)
        return cls(vocab, merges, is_bytelevel=_is_bytelevel_file(path))

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if self.is_bytelevel:
            ids = encode_text_bytelevel(text, self.vocab, self.rank, self.unk_id)
        else:
            ids = encode_text(text, self.vocab, self.rank, self.unk_id)
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        if self.is_bytelevel:
            return decode_ids_bytelevel(ids, self.inv_vocab, skip_specials=skip_special_tokens)
        return decode_ids(ids, self.inv_vocab, skip_specials=skip_special_tokens)

    def roundtrip(self, text: str) -> tuple[list[int], str, bool]:
        ids = self.encode(text)
        back = self.decode(ids)
        return ids, back, back == text

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


def train_tokenizer(
    input_path: str | Path,
    output_path: str | Path,
    vocab_size: int = 16000,
    min_frequency: int = 2,
    seed: int = 42,
) -> Path:
    set_all_seeds(seed)
    inp = resolve_path(input_path)
    out = resolve_path(output_path)
    files = _collect_files(inp)
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"input not found: {f} (run data/prepare.py first)")
    backend = "fallback-bpe"
    vocab: dict[str, int] = {}
    merges: list[str] = []
    try:
        vocab, merges = hf_train(files, vocab_size, min_frequency)
        backend = "hf-bpe"
    except ImportError:
        texts = _read_texts(files)
        vocab, merges = fallback_train(texts, vocab_size, min_frequency)
    except (ValueError, TypeError, RuntimeError, OSError) as e:
        print(f"hf train failed ({files}: {e}), using fallback", file=sys.stderr)
        texts = _read_texts(files)
        vocab, merges = fallback_train(texts, vocab_size, min_frequency)
    for t, i in SPECIAL_IDS.items():
        vocab[t] = i
    inv_check = sorted(vocab.items(), key=lambda kv: kv[1])
    fixed: dict[str, int] = {}
    for idx, (tok, _) in enumerate(inv_check):
        if tok in SPECIAL_IDS:
            continue
        fixed[tok] = idx
    for t, i in SPECIAL_IDS.items():
        fixed[t] = i
    order = sorted(fixed.items(), key=lambda kv: kv[1])
    renumbered: dict[str, int] = {}
    for new_id, (tok, _) in enumerate(order):
        renumbered[tok] = new_id
    vocab = renumbered
    pad_vocab_to_size(vocab, vocab_size)
    if len(vocab) != vocab_size:
        raise ValueError(f"vocab_size check failed: got {len(vocab)}, want {vocab_size}")
    pre_tok = "bytelevel" if backend == "hf-bpe" else "whitespace"
    save_tokenizer(out, vocab, merges, backend, pre_tok)
    tok = SimpleTokenizer(vocab, merges, is_bytelevel=(pre_tok == "bytelevel"))
    sample_texts = _read_texts(files)
    probe = ""
    for t in sample_texts:
        for line in t.splitlines():
            if line.strip():
                probe = line.strip()[:200]
                break
        if probe:
            break
    if not probe:
        probe = "hello world"
    _, back, ok = tok.roundtrip(probe)
    if not ok:
        print(f"warning: roundtrip mismatch on probe: {probe!r} -> {back!r}", file=sys.stderr)
    print(f"backend={backend} vocab_size={len(vocab)} merges={len(merges)} out={out}")
    return out


def main() -> None:
    cfg = Config()
    _ = cfg
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/sample.txt")
    p.add_argument("--output", default="tokenizer.json")
    p.add_argument("--vocab-size", type=int, default=16000)
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train_tokenizer(args.input, args.output, vocab_size=args.vocab_size, min_frequency=args.min_frequency, seed=args.seed)


if __name__ == "__main__":
    main()
