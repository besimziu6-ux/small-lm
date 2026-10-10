from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
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

try:
    from utils.seed import set_all_seeds
except ImportError:
    def set_all_seeds(seed, **kwargs):  # type: ignore[no-redef]
        random.seed(seed)

DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DEFAULT_SUBSET = "sample-10BT"
DEFAULT_SPLIT = "train"
SHARD_BYTES = 100 * 1024 * 1024


def _resolve_out_dir(out_dir: str | Path) -> Path:
    p = Path(out_dir)
    if p.is_absolute():
        return p
    return resolve_path(p)


def _normalize_doc(text: str) -> str:
    t = text.replace("\r", " ").replace("\n", " ").strip()
    return t


def _passes_score(row: dict, score_min: float) -> bool:
    if "score" not in row:
        return True
    s = row.get("score")
    if s is None:
        return True
    try:
        return float(s) >= score_min
    except (TypeError, ValueError):
        return True


def build_fineweb(
    out_dir: str | Path = "data/fineweb",
    max_gb: float = 3.0,
    score_min: float = 0.0,
    split: str = DEFAULT_SPLIT,
    subset: str = DEFAULT_SUBSET,
    seed: int = 42,
    force: bool = False,
) -> Path | None:
    try:
        set_all_seeds(seed)
    except RuntimeError:
        random.seed(seed)
    out = _resolve_out_dir(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    max_bytes = int(max_gb * 1024 * 1024 * 1024)

    if manifest_path.exists() and not force:
        try:
            man = json.loads(manifest_path.read_text(encoding="utf-8"))
            shards = man.get("shards", [])
            if shards and all((out / s["file"]).exists() for s in shards):
                if man.get("max_bytes") == max_bytes and man.get("score_min") == score_min:
                    print(f"cached: {manifest_path} ({len(shards)} shards)")
                    return manifest_path
        except (ValueError, KeyError, OSError) as e:
            print(f"manifest re-read failed ({manifest_path}: {e}), rebuilding", file=sys.stderr)

    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets lib missing, run `pip install datasets` to fetch FineWeb-Edu", file=sys.stderr)
        return None

    try:
        try:
            ds = load_dataset(DATASET_NAME, subset, split=split, streaming=True)
        except Exception:
            ds = load_dataset(DATASET_NAME, split=split, streaming=True)
    except Exception as e:
        print(f"fineweb-edu unavailable ({e})", file=sys.stderr)
        return None

    docs: list[str] = []
    total_bytes = 0
    try:
        for row in ds:
            if total_bytes >= max_bytes:
                break
            if isinstance(row, dict):
                if not _passes_score(row, score_min):
                    continue
                raw = row.get("text", "")
                text = _normalize_doc(str(raw) if raw is not None else "")
            else:
                text = _normalize_doc(str(row))
            if not text:
                continue
            lb = len((text + "\n").encode("utf-8", errors="ignore"))
            if total_bytes + lb > max_bytes and docs:
                break
            docs.append(text)
            total_bytes += lb
    except Exception as e:
        print(f"fineweb stream failed ({e})", file=sys.stderr)
        if not docs:
            return None

    shards: list[str] = []
    cur: list[str] = []
    cur_bytes = 0
    for doc in docs:
        line = doc + "\n"
        lb = len(line.encode("utf-8", errors="ignore"))
        if cur and cur_bytes + lb > SHARD_BYTES and cur_bytes >= SHARD_BYTES // 2:
            shards.append("".join(cur))
            cur = []
            cur_bytes = 0
        cur.append(line)
        cur_bytes += lb
    if cur:
        shards.append("".join(cur))
    if not shards and docs:
        shards = ["".join(doc + "\n" for doc in docs)]

    for old in sorted(out.glob("shard_*.txt")):
        old.unlink()

    shard_infos = []
    total_chars = 0
    total_disk = 0
    for i, content in enumerate(shards):
        name = f"shard_{i:04d}.txt"
        p = out / name
        p.write_text(content, encoding="utf-8")
        nbytes = p.stat().st_size
        nchars = len(content)
        total_chars += nchars
        total_disk += nbytes
        shard_infos.append({"file": name, "bytes": nbytes, "chars": nchars, "approx_tokens": nchars // 4})

    manifest = {
        "dataset": DATASET_NAME,
        "subset": subset,
        "split": split,
        "docs": len(docs),
        "num_docs": len(docs),
        "num_shards": len(shard_infos),
        "shards": shard_infos,
        "bytes": total_disk,
        "total_bytes": total_disk,
        "total_chars": total_chars,
        "rough_tokens": total_chars // 4,
        "total_approx_tokens": total_chars // 4,
        "shard_bytes": SHARD_BYTES,
        "max_bytes": max_bytes,
        "score_min": score_min,
        "seed": seed,
        "note": "shards are raw text, one doc per line; tokenization stays in dataset.py (chars/4 rough token estimate)",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote: {out} ({len(shard_infos)} shards, {len(docs)} docs, {total_disk} bytes, ~{total_chars // 4} tokens)")
    return manifest_path


def main() -> None:
    p = argparse.ArgumentParser(description="FineWeb-Edu fetch pipeline (raw-text shards, one doc per line)")
    p.add_argument("--output-dir", default="data/fineweb")
    p.add_argument("--max-gb", type=float, default=3.0, help="cap total output gigabytes (default: %(default)s)")
    p.add_argument("--score-min", type=float, default=0.0, help="min score column value (default: %(default)s)")
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--subset", default=DEFAULT_SUBSET)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    out = build_fineweb(
        out_dir=args.output_dir,
        max_gb=args.max_gb,
        score_min=args.score_min,
        split=args.split,
        subset=args.subset,
        seed=args.seed,
        force=args.force,
    )
    if out is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
