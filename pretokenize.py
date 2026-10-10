from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from utils.paths import resolve_path
except ImportError:
    def resolve_path(p):  # type: ignore[no-redef]
        q = Path(p)
        if q.is_absolute():
            return q
        if Path.cwd().name == "small-lm":
            return Path.cwd() / q
        return SCRIPT_DIR / q

MAX_DOC_TOKENS = 1024


def _split_docs(text: str) -> list[str]:
    docs = [d.strip() for d in text.split("\n") if d.strip()]
    if len(docs) >= 10:
        return docs
    chunks = [text[i:i + 2000] for i in range(0, len(text), 2000)]
    chunks = [c.strip() for c in chunks if c.strip()]
    return chunks or [text.strip()]


def _collect_inputs(inp: Path) -> list[Path]:
    if inp.is_file():
        return [inp]
    if inp.is_dir():
        shards = sorted(inp.glob("shard_*.txt"))
        if shards:
            return shards
        files = sorted(inp.glob("*.txt"))
        if files:
            return files
        return sorted(inp.rglob("*.txt"))
    raise FileNotFoundError(f"input not found: {inp}")


def _resolve_tokenizer(req: str) -> Path:
    p = resolve_path(req)
    if p.exists():
        return p
    if Path(req).name in ("tokenizer32k.json", "tokenizer_wiki32k.json", "tokenizer.json"):
        for cand in ("tokenizer32k.json", "tokenizer_wiki32k.json", "tokenizer.json"):
            q = resolve_path(cand)
            if q.exists():
                return q
    raise FileNotFoundError(f"tokenizer not found: {p} (tried fallback tokenizer32k.json/tokenizer_wiki32k.json)")


def _is_hf_format(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and "model" in data


def _is_bytelevel_payload(data: dict) -> bool:
    if data.get("pre_tokenizer") == "bytelevel":
        return True
    pt = data.get("pre_tokenizer")
    if isinstance(pt, dict) and pt.get("type") == "ByteLevel":
        return True
    return False


def _merges_to_tuples(raw_merges: list) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for m in raw_merges:
        if isinstance(m, (list, tuple)):
            if len(m) == 2:
                out.append((str(m[0]), str(m[1])))
            continue
        s = str(m)
        a, sep, b = s.partition(" ")
        if sep and a and b:
            out.append((a, b))
    return out


def _load_encoder(tok_path: Path):
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers import pre_tokenizers
    except ImportError:
        from train_tokenizer import SimpleTokenizer

        tok = SimpleTokenizer.from_file(tok_path)
        single = lambda text: tok.encode(text)  # noqa: E731
        return single, None, int(tok.bos_id), int(tok.eos_id), int(len(tok.vocab)), False
    try:
        data = json.loads(tok_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"tokenizer read failed ({tok_path}: {e}), using SimpleTokenizer", file=sys.stderr)
        from train_tokenizer import SimpleTokenizer

        tok = SimpleTokenizer.from_file(tok_path)
        single = lambda text: tok.encode(text)  # noqa: E731
        return single, None, int(tok.bos_id), int(tok.eos_id), int(len(tok.vocab)), False
    if isinstance(data, dict) and "vocab" in data and "merges" in data:
        try:
            vocab = {str(k): int(v) for k, v in data["vocab"].items()}
            merges = _merges_to_tuples(data.get("merges", []))
            is_bl = _is_bytelevel_payload(data)
            hf = Tokenizer(BPE(vocab=vocab, merges=merges, unk_token="<unk>"))
            if is_bl:
                hf.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            else:
                hf.pre_tokenizer = pre_tokenizers.Whitespace()

            bos = int(vocab.get("<bos>", 2))
            eos = int(vocab.get("<eos>", 3))
            vs = len(vocab)

            def encode(text: str) -> list[int]:
                return hf.encode(text, add_special_tokens=False).ids

            def encode_batch(texts: list[str]) -> list[list[int]]:
                return [e.ids for e in hf.encode_batch(texts, add_special_tokens=False)]

            return encode, encode_batch, bos, eos, vs, True
        except (ValueError, RuntimeError, OSError) as e:
            print(f"hf in-memory build failed ({tok_path}: {e}), using SimpleTokenizer", file=sys.stderr)
    elif _is_hf_format(tok_path):
        try:
            hf = Tokenizer.from_file(str(tok_path))
            vocab = hf.get_vocab()
            bos = int(vocab.get("<bos>", 2))
            eos = int(vocab.get("<eos>", 3))
            vs = int(hf.get_vocab_size())

            def encode(text: str) -> list[int]:
                return hf.encode(text, add_special_tokens=False).ids

            def encode_batch(texts: list[str]) -> list[list[int]]:
                return [e.ids for e in hf.encode_batch(texts, add_special_tokens=False)]

            return encode, encode_batch, bos, eos, vs, True
        except ImportError:
            pass
        except (ValueError, RuntimeError, OSError) as e:
            print(f"hf tokenizer load failed ({tok_path}: {e}), using SimpleTokenizer", file=sys.stderr)
    from train_tokenizer import SimpleTokenizer

    tok = SimpleTokenizer.from_file(tok_path)
    return (lambda text: tok.encode(text)), None, int(tok.bos_id), int(tok.eos_id), int(len(tok.vocab)), False


def _write_bin(path: Path, ids: list[int], dtype: str) -> None:
    if dtype not in ("uint16", "uint32"):
        raise ValueError(f"unknown dtype {dtype!r}, want 'uint16' or 'uint32'")
    try:
        import numpy as np

        dt = np.dtype("<u2") if dtype == "uint16" else np.dtype("<u4")
        np.array(ids, dtype=dt).tofile(str(path))
        return
    except ImportError:
        pass
    import array
    import sys as _sys

    a = array.array("H" if dtype == "uint16" else "I", ids)
    if _sys.byteorder != "little":
        a.byteswap()
    with open(path, "wb") as f:
        a.tofile(f)


def pretokenize(
    input_path: str | Path = "data/wiki",
    tokenizer_path: str | Path = "tokenizer32k.json",
    output_dir: str | Path = "data/ids",
    dtype: str = "uint16",
    max_doc_tokens: int = MAX_DOC_TOKENS,
) -> Path:
    if dtype not in ("uint16", "uint32"):
        raise ValueError(f"unknown dtype {dtype!r}, want 'uint16' or 'uint32'")
    inp = resolve_path(input_path)
    tok_path = _resolve_tokenizer(str(tokenizer_path))
    out = resolve_path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = _collect_inputs(inp)
    if not files:
        raise FileNotFoundError(f"no .txt shards under {inp}")
    encode, encode_batch, bos, eos, vocab_size, fast = _load_encoder(tok_path)
    mode = "hf-rust" if fast else "SimpleTokenizer"
    print(f"tokenizer={tok_path} vocab_size={vocab_size} mode={mode} dtype={dtype}")
    try:
        first_text = files[0].read_text(encoding="utf-8", errors="ignore")
        sample = first_text[: 2 * 1024 * 1024]
        if sample.strip():
            probe_ids = encode(sample)
            if probe_ids:
                nbytes = len(sample.encode("utf-8"))
                print(f"probe bytes/token={nbytes / len(probe_ids):.2f} (sample {nbytes} bytes -> {len(probe_ids)} ids)")
            else:
                print("probe bytes/token=N/A (empty sample)")
        else:
            print("probe bytes/token=N/A (empty first shard)")
    except Exception as e:
        print(f"probe failed: {e}", file=sys.stderr)
    shard_ids: list[list[int]] = []
    global_max = -1
    for i, src in enumerate(files):
        text = src.read_text(encoding="utf-8", errors="ignore")
        docs = _split_docs(text)
        if encode_batch is not None:
            try:
                batch = encode_batch(docs)
            except Exception:
                batch = [encode(d) for d in docs]
        else:
            batch = [encode(d) for d in docs]
        ids: list[int] = []
        for enc in batch:
            enc = enc[:max_doc_tokens]
            if enc:
                m = max(enc)
                if m > global_max:
                    global_max = m
            ids.append(bos)
            ids.extend(enc)
            ids.append(eos)
        shard_ids.append(ids)
        print(f"[{i + 1}/{len(files)}] {src.name}: {len(docs)} docs -> {len(ids)} ids")
    if global_max >= 0 and dtype == "uint16" and global_max >= 65536:
        raise ValueError(f"max id {global_max} >= 65536 exceeds uint16; re-run with --dtype uint32")
    total = 0
    infos = []
    for i, (src, ids) in enumerate(zip(files, shard_ids)):
        name = f"ids_{i:04d}.bin"
        _write_bin(out / name, ids, dtype)
        total += len(ids)
        infos.append({"file": name, "ids": len(ids), "source": src.name})
    manifest = {
        "dtype": dtype,
        "vocab_size": vocab_size,
        "total_ids": total,
        "shards": infos,
        "tokenizer": tok_path.name,
    }
    man_path = out / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote: {out} ({len(infos)} shards, {total} ids, dtype={dtype})")
    return man_path


def main() -> None:
    p = argparse.ArgumentParser(description="pre-tokenize text shards to uint16/uint32 id bins")
    p.add_argument("--input", default="data/wiki")
    p.add_argument("--tokenizer", default="tokenizer32k.json")
    p.add_argument("--output-dir", default="data/ids")
    p.add_argument("--dtype", default="uint16", choices=["uint16", "uint32"])
    args = p.parse_args()
    pretokenize(args.input, args.tokenizer, args.output_dir, dtype=args.dtype)


if __name__ == "__main__":
    main()
