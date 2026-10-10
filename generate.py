from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch
import torch.nn.functional as F

from config import Config, resolve_device
from utils.paths import resolve_path
from utils.preset import infer_preset_and_seq
from utils.seed import set_all_seeds, wants_deterministic


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate text")
    p.add_argument("--checkpoint", default="checkpoints/best_smoke.pt",
                   help="path to checkpoint (default: %(default)s)")
    p.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"],
                   help="device to use (default: %(default)s)")
    p.add_argument("--prompt", default="Once upon")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preset", default=None)
    p.add_argument("--tokenizer", default="tokenizer.json")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args(argv)


def exit_missing_checkpoint(ckpt_path: Path) -> None:
    parent = ckpt_path.parent if str(ckpt_path.parent) not in ("", ".") else Path("checkpoints")
    try:
        if parent.is_dir():
            names = sorted(x.name for x in parent.iterdir() if x.is_file())
            listing = ", ".join(names) if names else "(empty)"
        else:
            listing = "(missing dir)"
    except OSError:
        listing = "(unreadable)"
    print(f"checkpoint not found: {ckpt_path}; contents of {parent}: {listing}; try `ls checkpoints/`", file=sys.stderr)
    raise SystemExit(1)


def validate_args(args) -> None:
    if args.max_new_tokens is not None and args.max_new_tokens < 0:
        print(f"error: --max-new-tokens must be >= 0 (got {args.max_new_tokens})", file=sys.stderr)
        raise SystemExit(2)
    try:
        temp_nan = args.temperature is not None and math.isnan(float(args.temperature))
    except (TypeError, ValueError):
        temp_nan = False
    if temp_nan:
        print(f"error: --temperature must not be NaN (got {args.temperature}); use e.g. --temperature 0.0 for greedy", file=sys.stderr)
        raise SystemExit(2)
    if args.top_k is not None and args.top_k < 0:
        print(f"error: --top-k must be >= 0 (got {args.top_k})", file=sys.stderr)
        raise SystemExit(2)
    if args.top_p is not None and not (0.0 <= args.top_p <= 1.0):
        print(f"error: --top-p must be in [0.0, 1.0] (got {args.top_p})", file=sys.stderr)
        raise SystemExit(2)


def sample_next_id(logits_1d: torch.Tensor, temperature: float, top_k: int, top_p: float) -> int:
    if temperature is None or temperature <= 0 or temperature < 1e-9:
        return int(torch.argmax(logits_1d).item())
    logits = logits_1d.float() / temperature
    vocab = logits.numel()
    if top_k is not None and top_k > 0 and top_k < vocab:
        vals, idx = torch.topk(logits, top_k)
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(0, idx, vals)
        logits = filtered
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        cutoff = cum > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_logits[cutoff] = float("-inf")
        logits = torch.empty_like(logits)
        logits.scatter_(0, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def main(argv=None) -> None:
    args = parse_args(argv)
    validate_args(args)

    gc = Config()
    gc.seed = args.seed
    gc.threads = args.threads
    gc.device = resolve_device(args.device)
    device = gc.device
    try:
        gc.apply_torch_settings()
    except RuntimeError as e:
        print(f"warning: torch settings failed ({e}), using defaults", file=sys.stderr)
        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(args.threads)
        except RuntimeError:
            pass
    torch.set_default_dtype(torch.float32)
    try:
        set_all_seeds(args.seed, threads=args.threads, deterministic=wants_deterministic(device))
    except RuntimeError as e:
        print(f"warning: deterministic seeding failed ({e}), continuing non-deterministically", file=sys.stderr)
        torch.manual_seed(args.seed)
    if device == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(args.seed)
        except Exception as e:
            print(f"warning: cuda seeding failed ({e})", file=sys.stderr)

    ckpt_path = resolve_path(args.checkpoint)
    if not ckpt_path.exists():
        exit_missing_checkpoint(ckpt_path)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        import sys as _sys

        print(f"warning: weights_only load failed for {ckpt_path}, retrying trusted local load", file=_sys.stderr)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    preset, seq_len = infer_preset_and_seq(
        ckpt if isinstance(ckpt, dict) else {}, args.preset, args.seq_len, ckpt_path, default_seq=256
    )

    from model import build_model

    model = build_model(preset=preset, seed=args.seed, base_cfg=gc)
    model.to(torch.float32)
    model.to(device)
    model.eval()
    vocab_size = int(model.cfg.vocab_size)
    max_ctx = int(model.cfg.max_seq_len)

    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        model.load_state_dict(state, strict=False)
    model.eval()

    from train_tokenizer import SimpleTokenizer

    tok_path = resolve_path(args.tokenizer)
    tok = SimpleTokenizer.from_file(tok_path)
    pad_id = int(tok.pad_id)
    eos_id = int(tok.eos_id)
    unk_id = int(tok.unk_id)
    _ = pad_id

    prompt_ids = tok.encode(args.prompt)
    if len(prompt_ids) == 0:
        prompt_ids = [int(tok.bos_id)]

    ctx = [int(i) if int(i) < vocab_size else unk_id for i in prompt_ids]
    ctx = ctx[-max_ctx:] if len(ctx) > max_ctx else ctx

    new_ids: list[int] = []
    model.eval()
    with torch.no_grad():
        for _ in range(max(0, int(args.max_new_tokens))):
            window = ctx[-max_ctx:]
            x = torch.tensor([window], dtype=torch.long, device=device)
            logits = model(x)
            last = logits[0, -1, :]
            nid = sample_next_id(last, args.temperature, args.top_k, args.top_p)
            if nid >= len(tok.vocab):
                nid = unk_id
            new_ids.append(int(nid))
            ctx.append(int(nid) if int(nid) < vocab_size else unk_id)
            if int(nid) == eos_id:
                break

    text = tok.decode(new_ids)
    print(f"prompt={args.prompt!r}")
    print(f"new_token_ids={new_ids}")
    print(f"num_new_tokens={len(new_ids)}")
    print(f"text={text}")
    print(text)


if __name__ == "__main__":
    main()
