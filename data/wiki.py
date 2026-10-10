from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

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

WT103_S3_ZIP_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
WT103_S3_MEMBERS = [
    ("train", "wikitext-103-raw/wiki.train.tokens"),
    ("valid", "wikitext-103-raw/wiki.valid.tokens"),
    ("test", "wikitext-103-raw/wiki.test.tokens"),
]

WT103_URLS = [
    ("train", "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/train.txt"),
    ("valid", "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/valid.txt"),
    ("test", "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/test.txt"),
]

FALLBACK_URLS = [
    ("train", "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"),
    ("valid", "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt"),
]

DEFAULT_SHARD_MB = 50
SYNTHETIC_FALLBACK_BYTES = 5 * 1024 * 1024


def _fetch_url(url: str, timeout: int = 30) -> str | None:
    try:
        req = Request(url, headers={"User-Agent": "small-lm/1.0"})
        with urlopen(req, timeout=timeout) as r:
            raw = r.read()
        return raw.decode("utf-8", errors="ignore")
    except (OSError, ValueError) as e:
        print(f"fetch failed ({url}: {e})", file=sys.stderr)
        return None


def _fetch_bytes(url: str, timeout: int = 30) -> bytes | None:
    try:
        req = Request(url, headers={"User-Agent": "small-lm/1.0"})
        with urlopen(req, timeout=timeout) as r:
            return r.read()
    except (OSError, ValueError) as e:
        print(f"fetch failed ({url}: {e})", file=sys.stderr)
        return None


def _clean_wikitext_text(text: str) -> str:
    kept: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith(" = "):
            continue
        if line.strip().startswith("="):
            continue
        kept.append(line)
    out = "\n".join(kept)
    return out + "\n" if out else ""


def _load_datasets_wikitext103() -> list[tuple[str, str]] | None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets not installed, run `pip install datasets` for WikiText-103 primary source", file=sys.stderr)
        return None
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    except Exception as e:
        print(f"datasets wikitext-103 unavailable ({e})", file=sys.stderr)
        return None
    try:
        wanted = [("train", "train"), ("validation", "valid"), ("test", "test")]
        parts: list[tuple[str, str]] = []
        for ds_split, out_split in wanted:
            if ds_split not in ds:
                continue
            raw = "\n".join(str(r.get("text", "") if isinstance(r, dict) else r) for r in ds[ds_split])
            cleaned = _clean_wikitext_text(raw)
            if len(cleaned.strip()) >= 500:
                parts.append((out_split, cleaned))
        return parts or None
    except Exception as e:
        print(f"datasets wikitext-103 parse failed ({e})", file=sys.stderr)
        return None


def _load_s3_zip(out: Path, force: bool = False, timeout: int = 30) -> list[tuple[str, str]] | None:
    zip_path = out / "wikitext-103-raw-v1.zip"
    try:
        out.mkdir(parents=True, exist_ok=True)
        extracted = [(split, out / member) for split, member in WT103_S3_MEMBERS]
        if not force and all(p.exists() and p.stat().st_size > 0 for _, p in extracted):
            parts: list[tuple[str, str]] = []
            for split, p in extracted:
                parts.append((split, _clean_wikitext_text(p.read_text(encoding="utf-8", errors="ignore"))))
            if all(len(t.strip()) >= 1000 for _, t in parts):
                return parts
        if not (zip_path.exists() and zip_path.stat().st_size > 0 and not force):
            blob = _fetch_bytes(WT103_S3_ZIP_URL, timeout=timeout)
            if not blob:
                return None
            zip_path.write_bytes(blob)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out)
        parts = []
        for split, member in WT103_S3_MEMBERS:
            p = out / member
            if not p.exists() or p.stat().st_size == 0:
                return None
            text = _clean_wikitext_text(p.read_text(encoding="utf-8", errors="ignore"))
            if len(text.strip()) < 500:
                return None
            parts.append((split, text))
        return parts
    except (OSError, ValueError, zipfile.BadZipFile) as e:
        print(f"s3 wikitext-103 zip unavailable ({e})", file=sys.stderr)
        return None


def _synthetic_corpus(seed: int = 42, target_bytes: int = SYNTHETIC_FALLBACK_BYTES) -> str:
    rng = random.Random(seed)
    subjects = ["the cat", "a dog", "the child", "a robot", "the bird", "a farmer", "the teacher", "a king", "the river", "a star"]
    verbs = ["runs", "walks", "sings", "reads", "writes", "builds", "dreams", "paints", "counts", "discovers"]
    objects = ["in the garden", "under the sky", "near the river", "with friends", "at dawn", "in the city", "on the hill", "by the sea", "in silence", "with joy"]
    extra_words = ["apple", "bridge", "cloud", "dance", "engine", "forest", "grain", "harbor", "island", "jungle", "kite", "lantern", "meadow", "needle", "ocean", "piano", "quartz", "rainbow", "stone", "tower", "umbrella", "valley", "window", "yarn", "zebra", "number", "history", "science", "music", "light"]
    lines: list[str] = []
    total = 0
    i = 0
    while total < target_bytes:
        s = rng.choice(subjects)
        v = rng.choice(verbs)
        o = rng.choice(objects)
        n = rng.randint(0, 9999)
        w1 = rng.choice(extra_words)
        w2 = rng.choice(extra_words)
        if i % 7 == 0:
            line = f"Once upon a time {s} {v} {o}. It found {w1} and {w2} on day {n}."
        elif i % 7 == 1:
            line = f"{s.capitalize()} {v} {o} with {w1} {w2} {n} times."
        elif i % 7 == 2:
            line = f"In wikitext style: {w1} {w2} = {n} ; {s} {v} {o}."
        else:
            line = f"{s} {v} {o}, {w1} {w2} {n}."
        lines.append(line)
        total += len(line) + 1
        i += 1
    return "\n".join(lines) + "\n"


def _resolve_out_dir(out_dir: str | Path) -> Path:
    p = Path(out_dir)
    if p.is_absolute():
        return p
    return resolve_path(p)


def _cap_bytes(text: str, max_bytes: int | None) -> str:
    if max_bytes is None:
        return text
    raw = text.encode("utf-8", errors="ignore")[:max_bytes].decode("utf-8", errors="ignore")
    return raw


def _split_shards(text: str, shard_bytes: int) -> list[str]:
    lines = text.splitlines(keepends=True)
    shards: list[str] = []
    cur: list[str] = []
    cur_bytes = 0
    for line in lines:
        lb = len(line.encode("utf-8", errors="ignore"))
        if cur and cur_bytes + lb > shard_bytes and cur_bytes >= shard_bytes // 2:
            shards.append("".join(cur))
            cur = []
            cur_bytes = 0
        cur.append(line)
        cur_bytes += lb
    if cur:
        shards.append("".join(cur))
    if not shards:
        shards = [text] if text else [""]
    return shards


def build_wiki(
    out_dir: str | Path = "data/wiki",
    max_mb: float | None = None,
    shard_mb: float = DEFAULT_SHARD_MB,
    seed: int = 42,
    force: bool = False,
    timeout: int = 30,
) -> Path:
    try:
        set_all_seeds(seed)
    except RuntimeError:
        random.seed(seed)
    out = _resolve_out_dir(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"

    max_bytes = int(max_mb * 1024 * 1024) if max_mb is not None else None
    shard_bytes = max(int(shard_mb * 1024 * 1024), 256 * 1024)

    if manifest_path.exists() and not force:
        try:
            man = json.loads(manifest_path.read_text(encoding="utf-8"))
            shards = man.get("shards", [])
            if shards and all((out / s["file"]).exists() for s in shards):
                want_max = int(max_mb * 1024 * 1024) if max_mb is not None else None
                if man.get("max_bytes") == want_max and man.get("shard_bytes") == shard_bytes:
                    print(f"cached: {manifest_path} ({len(shards)} shards)")
                    return manifest_path
        except (ValueError, KeyError, OSError) as e:
            print(f"manifest re-read failed ({manifest_path}: {e}), rebuilding", file=sys.stderr)

    raw_parts: list[tuple[str, str]] = _load_datasets_wikitext103() or []
    if len(raw_parts) < len(WT103_URLS):
        s3_parts = _load_s3_zip(out, force=force, timeout=timeout)
        if s3_parts:
            have_s3 = {s for s, _ in raw_parts}
            for s, t in s3_parts:
                if s not in have_s3:
                    raw_parts.append((s, t))
        if len(raw_parts) < len(WT103_URLS):
            for split, url in WT103_URLS:
                if any(s == split for s, _ in raw_parts):
                    continue
                cache_path = out / f"{split}.txt"
                if cache_path.exists() and cache_path.stat().st_size > 0 and not force:
                    raw_parts.append((split, cache_path.read_text(encoding="utf-8", errors="ignore")))
                    continue
                text = _fetch_url(url, timeout=timeout)
                if text and len(text.strip()) >= 1000:
                    cache_path.write_text(text, encoding="utf-8")
                    raw_parts.append((split, text))
                else:
                    print(f"primary {split} unavailable, trying fallback", file=sys.stderr)

    if len(raw_parts) < len(WT103_URLS):
        have = {s for s, _ in raw_parts}
        for split, url in FALLBACK_URLS:
            if split in have:
                continue
            cache_path = out / f"{split}.txt"
            if cache_path.exists() and cache_path.stat().st_size > 0 and not force:
                raw_parts.append((split, cache_path.read_text(encoding="utf-8", errors="ignore")))
                continue
            text = _fetch_url(url, timeout=timeout)
            if text and len(text.strip()) >= 500:
                cache_path.write_text(text, encoding="utf-8")
                raw_parts.append((split, text))

    order = {"train": 0, "valid": 1, "test": 2}
    raw_parts.sort(key=lambda kv: order.get(kv[0], 99))

    if raw_parts:
        combined = "\n\n".join(t for _, t in raw_parts)
        if not combined.endswith("\n"):
            combined += "\n"
        if max_bytes is not None:
            combined = _cap_bytes(combined, max_bytes)
        if len(combined.strip()) < 1000:
            target = max_bytes if max_bytes is not None else SYNTHETIC_FALLBACK_BYTES
            combined = _synthetic_corpus(seed=seed, target_bytes=target)
    else:
        target = max_bytes if max_bytes is not None else SYNTHETIC_FALLBACK_BYTES
        combined = _synthetic_corpus(seed=seed, target_bytes=target)

    shards = _split_shards(combined, shard_bytes)

    for old in sorted(out.glob("shard_*.txt")):
        old.unlink()

    shard_infos = []
    total_chars = 0
    total_bytes = 0
    for i, content in enumerate(shards):
        name = f"shard_{i:04d}.txt"
        p = out / name
        p.write_text(content, encoding="utf-8")
        nbytes = p.stat().st_size
        nchars = len(content)
        approx = nchars // 4
        total_chars += nchars
        total_bytes += nbytes
        shard_infos.append({"file": name, "bytes": nbytes, "chars": nchars, "approx_tokens": approx})

    manifest = {
        "dataset": "wikitext-103",
        "splits": [s for s, _ in raw_parts] if raw_parts else ["synthetic"],
        "num_shards": len(shard_infos),
        "shards": shard_infos,
        "total_bytes": total_bytes,
        "total_chars": total_chars,
        "total_approx_tokens": total_chars // 4,
        "shard_bytes": shard_bytes,
        "max_bytes": max_bytes,
        "seed": seed,
        "note": "shards are raw text; tokenization stays in dataset.py (chars/4 rough token estimate)",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote: {out} ({len(shard_infos)} shards, {total_bytes} bytes, ~{total_chars // 4} tokens)")
    return manifest_path


def main() -> None:
    p = argparse.ArgumentParser(description="WikiText-103 dataset pipeline (raw-text shards)")
    p.add_argument("--out-dir", default="data/wiki")
    p.add_argument("--max-mb", type=float, default=None, help="cap total output megabytes (e.g. 5 for small tests)")
    p.add_argument("--shard-mb", type=float, default=DEFAULT_SHARD_MB, help="target megabytes per shard (default: %(default)s)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    p.add_argument("--timeout", type=int, default=30)
    args = p.parse_args()
    build_wiki(args.out_dir, max_mb=args.max_mb, shard_mb=args.shard_mb, seed=args.seed, force=args.force, timeout=args.timeout)


if __name__ == "__main__":
    main()
