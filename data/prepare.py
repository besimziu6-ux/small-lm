from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from urllib.request import Request, urlopen

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import Config
from utils.seed import set_all_seeds

CANDIDATE_URLS = [
    "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
    "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
]

DEFAULT_MAX_BYTES = 5 * 1024 * 1024


def _fetch_url(url: str, timeout: int = 30) -> str | None:
    try:
        req = Request(url, headers={"User-Agent": "small-lm/1.0"})
        with urlopen(req, timeout=timeout) as r:
            raw = r.read()
        return raw.decode("utf-8", errors="ignore")
    except (OSError, ValueError) as e:
        print(f"fetch failed ({url}: {e}), using synthetic corpus", file=sys.stderr)
        return None


def _synthetic_corpus(seed: int = 42, target_bytes: int = 1_000_000) -> str:
    rng = random.Random(seed)
    subjects = ["the cat", "a dog", "the child", "a robot", "the bird", "a farmer", "the teacher", "a king", "the river", "a star"]
    verbs = ["runs", "walks", "sings", "reads", "writes", "builds", "dreams", "paints", "counts", "discovers"]
    objects = ["in the garden", "under the sky", "near the river", "with friends", "at dawn", "in the city", "on the hill", "by the sea", "in silence", "with joy"]
    extra_words = ["apple", "bridge", "cloud", "dance", "engine", "forest", "grain", "harbor", "island", "jungle", "kite", "lantern", "meadow", "needle", "ocean", "piano", "quartz", "rainbow", "stone", "tower", "umbrella", "valley", "window", "yarn", "zebra", "number", "history", "science", "music", "light"]
    for i in range(32, 127):
        extra_words.append(chr(i) * 1)
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


def prepare(output: Path, max_bytes: int = DEFAULT_MAX_BYTES, seed: int = 42, force: bool = False) -> Path:
    set_all_seeds(seed)
    output = Path(output)
    if not output.is_absolute():
        cwd_candidate = Path.cwd() / output
        script_candidate = SCRIPT_DIR / output
        if cwd_candidate.exists() and not force:
            pass
        elif script_candidate.parent.exists() or not cwd_candidate.parent.exists():
            if not str(output).startswith("data/"):
                output = script_candidate
            else:
                output = Path.cwd() / output if Path.cwd().name == "small-lm" else script_candidate
        else:
            output = cwd_candidate
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0 and not force:
        print(f"cached: {output} ({output.stat().st_size} bytes)")
        return output
    chunks: list[str] = []
    total = 0
    for url in CANDIDATE_URLS:
        text = _fetch_url(url)
        if not text:
            continue
        take = text[: max(0, max_bytes - total)]
        chunks.append(take)
        total += len(take.encode("utf-8", errors="ignore"))
        if total >= max_bytes:
            break
    if chunks:
        combined = "".join(chunks)
        if len(combined.encode("utf-8", errors="ignore")) > max_bytes:
            raw = combined.encode("utf-8", errors="ignore")[:max_bytes].decode("utf-8", errors="ignore")
        else:
            raw = combined
        if len(raw.strip()) < 1000:
            raw = _synthetic_corpus(seed=seed)
    else:
        raw = _synthetic_corpus(seed=seed)
    output.write_text(raw, encoding="utf-8")
    print(f"wrote: {output} ({output.stat().st_size} bytes)")
    return output


def main() -> None:
    cfg = Config()
    default_out = cfg.data_dir / "raw" / "sample.txt"
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=str(default_out))
    p.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    prepare(Path(args.output), max_bytes=args.max_bytes, seed=args.seed, force=args.force)


if __name__ == "__main__":
    main()
