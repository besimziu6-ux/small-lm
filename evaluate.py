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
    p = argparse.ArgumentParser(description="Val perplexity (deterministic)")
    p.add_argument("--checkpoint", default="checkpoints/best_smoke.pt",
                   help="path to checkpoint (default: %(default)s)")
    p.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"],
                   help="device to use (default: %(default)s)")
    p.add_argument("--preset", default=None, help="smoke|base (default: infer from checkpoint, fallback smoke)")
    p.add_argument("--tokenizer", default="tokenizer.json")
    p.add_argument("--raw", default="data/raw/sample.txt")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
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


def main(argv=None) -> None:
    args = parse_args(argv)

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
        print(f"warning: weights_only load failed for {ckpt_path}, retrying trusted local load", file=sys.stderr)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    preset, seq_len = infer_preset_and_seq(
        ckpt if isinstance(ckpt, dict) else {}, args.preset, args.seq_len, ckpt_path
    )

    from model import build_model

    model = build_model(preset=preset, seed=args.seed, base_cfg=gc)
    model.to(torch.float32)
    model.to(device)
    model.eval()
    vocab_size = int(model.cfg.vocab_size)

    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        model.load_state_dict(state, strict=False)
    model.eval()

    from train_tokenizer import SimpleTokenizer

    tok_path = resolve_path(args.tokenizer)
    tok = SimpleTokenizer.from_file(tok_path)
    unk_id = int(tok.unk_id)

    from dataset import build_datasets

    raw_path = resolve_path(args.raw)
    _, val_ds = build_datasets(
        raw_path, tok_path, seq_len=seq_len, seed=args.seed, min_train_blocks=8
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if args.max_batches is not None and i >= args.max_batches:
                break
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                labels = labels.unsqueeze(0)
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            input_ids = torch.where(
                input_ids >= vocab_size, torch.tensor(unk_id, device=input_ids.device), input_ids
            )
            labels = torch.where(
                (labels != -100) & (labels >= vocab_size),
                torch.tensor(unk_id, dtype=labels.dtype, device=labels.device),
                labels,
            )
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )
            total += float(loss.item())
            count += 1

    if count == 0:
        print("no val batches", file=sys.stderr)
        raise SystemExit(1)
    avg = total / count
    ppl = math.exp(avg) if avg < 20 else float("inf")
    print(f"batches={count} val_loss={avg:.4f} perplexity={ppl:.4f}")


if __name__ == "__main__":
    main()
