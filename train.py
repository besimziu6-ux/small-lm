from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch
import torch.nn.functional as F

from config import Config, resolve_device
from utils.paths import resolve_path
from utils.preset import normalize_preset
from utils.seed import set_all_seeds, wants_deterministic


PRESET_DEFAULTS = {
    "smoke": {
        "seq_len": 128,
        "max_steps": 300,
        "micro_batch": 8,
        "effective_batch": 32,
        "eval_every": 50,
        "save_every": 200,
    },
    "base": {
        "seq_len": 256,
        "max_steps": 1000,
        "micro_batch": 4,
        "effective_batch": 32,
        "eval_every": 100,
        "save_every": 200,
    },
}

LR_DEFAULT = 3e-4
WD_DEFAULT = 0.1
BETAS_DEFAULT = (0.9, 0.95)
WARMUP_DEFAULT = 50
CLIP_DEFAULT = 1.0
SEED_DEFAULT = 42


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="train loop for small-lm")
    p.add_argument("--preset", default="smoke", choices=["smoke", "base"])
    p.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"],
                   help="device to use (default: %(default)s)")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None, help="effective batch (default 32)")
    p.add_argument("--micro-batch", type=int, default=None)
    p.add_argument("--lr", type=float, default=LR_DEFAULT)
    p.add_argument("--warmup-steps", type=int, default=WARMUP_DEFAULT)
    p.add_argument("--weight-decay", type=float, default=WD_DEFAULT)
    p.add_argument("--grad-clip", type=float, default=CLIP_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--raw", default="data/raw/sample.txt")
    p.add_argument("--tokenizer", default="tokenizer.json")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--max-val-batches", type=int, default=50)
    p.add_argument("--resume", nargs="?", const="auto", default=None,
                   help="resume from checkpoint: --resume [path] or --resume for latest")
    p.add_argument("--resume-from", default=None, help="explicit checkpoint path")
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args(argv)


def validate_args(args) -> None:
    if args.max_steps is not None and args.max_steps <= 0:
        print(f"error: --max-steps must be > 0 (got {args.max_steps})", file=sys.stderr)
        raise SystemExit(2)
    if args.seq_len is not None and args.seq_len < 8:
        print(f"error: --seq-len must be >= 8 (got {args.seq_len}); try --seq-len 128", file=sys.stderr)
        raise SystemExit(2)
    if args.micro_batch is not None and args.micro_batch < 1:
        print(f"error: --micro-batch must be >= 1 (got {args.micro_batch})", file=sys.stderr)
        raise SystemExit(2)
    if args.grad_clip is not None and not args.grad_clip > 0:
        print(f"error: --grad-clip must be > 0 (got {args.grad_clip})", file=sys.stderr)
        raise SystemExit(2)
    if args.warmup_steps is not None and args.warmup_steps < 0:
        print(f"error: --warmup-steps must be >= 0 (got {args.warmup_steps})", file=sys.stderr)
        raise SystemExit(2)


def lr_lambda_fn(step: int, warmup: int, max_steps: int) -> float:
    if max_steps <= 0:
        return 1.0
    if step < warmup and warmup > 0:
        return float(step + 1) / float(max(warmup, 1))
    if max_steps <= warmup:
        return 1.0
    prog = (step - warmup) / max(1, max_steps - warmup)
    prog = min(max(prog, 0.0), 1.0)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))


def get_rng_states():
    import numpy as np

    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    try:
        if torch.cuda.is_available():
            states["cuda"] = torch.cuda.get_rng_state_all()
    except Exception:
        pass
    return states


def set_rng_states(s) -> None:
    import numpy as np

    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if "cuda" in s:
        try:
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(s["cuda"])
        except (RuntimeError, KeyError, ValueError) as e:
            print(f"rng cuda restore failed ({e}), using fresh CUDA RNG", file=sys.stderr)


def find_latest_ckpt(ckpt_dir: Path, preset: str):
    cands = sorted(ckpt_dir.glob(f"ckpt_{preset}_step*.pt"))
    if cands:
        return cands[-1]
    last = ckpt_dir / f"last_{preset}.pt"
    if last.exists():
        return last
    generics = sorted(ckpt_dir.glob("ckpt_step*.pt")) + sorted(ckpt_dir.glob("*.pt"))
    return generics[-1] if generics else None


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def evaluate(model, val_loader, vocab_size: int, unk_id: int, max_batches: int, device: str = "cpu") -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                labels = labels.unsqueeze(0)
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            input_ids = torch.where(input_ids >= vocab_size, torch.tensor(unk_id, device=input_ids.device), input_ids)
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
    model.train()
    return total / max(count, 1)


def main(argv=None) -> None:
    args = parse_args(argv)
    validate_args(args)
    preset = normalize_preset(args.preset)
    d = PRESET_DEFAULTS[preset]

    seq_len = args.seq_len if args.seq_len is not None else d["seq_len"]
    max_steps = args.max_steps if args.max_steps is not None else d["max_steps"]
    eval_every = args.eval_every if args.eval_every is not None else d["eval_every"]
    save_every = args.save_every if args.save_every is not None else d["save_every"]
    effective = args.batch_size if args.batch_size is not None else d["effective_batch"]
    micro = args.micro_batch if args.micro_batch is not None else d["micro_batch"]
    if effective % micro != 0:
        raise ValueError(f"effective batch {effective} must be divisible by micro {micro}")
    accum = effective // micro

    gc = Config()
    gc.seed = args.seed
    gc.threads = args.threads
    gc.device = resolve_device(args.device)
    device = gc.device
    gc.ensure_dirs()
    try:
        gc.apply_torch_settings()
    except RuntimeError as e:
        print(f"warning: torch settings failed ({e}), using defaults", file=sys.stderr)
        torch.set_num_threads(gc.threads)
        try:
            torch.set_num_interop_threads(gc.threads)
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

    ckpt_dir = resolve_path(args.checkpoint_dir)
    log_dir = resolve_path(args.log_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    from model import build_model

    model = build_model(preset="smoke" if preset == "smoke" else "base", seed=args.seed, base_cfg=gc, init_settings=False)
    model.train()
    model.to(torch.float32)
    model.to(device)
    vocab_size = int(model.cfg.vocab_size)

    from dataset import build_datasets

    raw_path = resolve_path(args.raw)
    tok_path = resolve_path(args.tokenizer)
    need_blocks = max_steps * effective
    train_ds, val_ds = build_datasets(
        raw_path, tok_path, seq_len=seq_len, seed=args.seed,
        min_train_blocks=max(need_blocks, 8),
        deterministic=wants_deterministic(device),
    )

    from train_tokenizer import SimpleTokenizer

    try:
        tok = SimpleTokenizer.from_file(tok_path)
        unk_id = int(tok.unk_id)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"tokenizer load failed ({tok_path}: {e}), using unk_id=1", file=sys.stderr)
        unk_id = 1

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=micro, shuffle=True, num_workers=0, generator=g
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=micro, shuffle=False, num_workers=0
    )

    decay, no_decay = [], []
    for n, prm in model.named_parameters():
        if not prm.requires_grad:
            continue
        if prm.dim() >= 2:
            decay.append(prm)
        else:
            no_decay.append(prm)
    groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if not decay:
        groups = [{"params": no_decay, "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=BETAS_DEFAULT)

    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: lr_lambda_fn(s, args.warmup_steps, max_steps)
    )

    resume_path = args.resume_from
    if resume_path is None and args.resume is not None:
        if args.resume == "auto":
            found = find_latest_ckpt(ckpt_dir, preset)
            if found is None:
                print(f"no checkpoint found in {ckpt_dir}, starting fresh", file=sys.stderr)
            else:
                resume_path = str(found)
        else:
            resume_path = args.resume

    start_step = 0
    best_val = float("inf")
    if resume_path is not None:
        rp = resolve_path(resume_path)
        if not rp.exists():
            alt = ckpt_dir / Path(resume_path).name
            if alt.exists():
                rp = alt
        ckpt = None
        try:
            ckpt = torch.load(rp, map_location=device, weights_only=True)
        except Exception:
            print(f"warning: weights_only load failed for {rp}, retrying trusted local load", file=sys.stderr)
            ckpt = torch.load(rp, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        try:
            sched.load_state_dict(ckpt["scheduler_state"])
        except (RuntimeError, KeyError) as e:
            print(f"scheduler restore failed ({rp}: {e}), stepping manually", file=sys.stderr)
            for _ in range(int(ckpt.get("step", 0))):
                sched.step()
        start_step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("best_val", float("inf")))
        try:
            rng_payload = {"python": ckpt["rng_python"], "numpy": ckpt["rng_numpy"], "torch": ckpt["rng_torch"]}
            if "rng_cuda" in ckpt:
                rng_payload["cuda"] = ckpt["rng_cuda"]
            set_rng_states(rng_payload)
        except (RuntimeError, KeyError) as e:
            print(f"rng restore failed ({rp}: {e}), using fresh RNG", file=sys.stderr)
        model.train()
        model.to(device)
        print(f"resumed from {rp} at step {start_step} best_val={best_val:.4f}")

    if start_step >= max_steps:
        print(f"already at step {start_step} >= max_steps {max_steps}, nothing to do")
        return

    csv_path = log_dir / f"train_{preset}.csv"
    write_header = not (csv_path.exists() and start_step > 0 and csv_path.stat().st_size > 0)
    csv_file = open(csv_path, "a" if start_step > 0 else "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=["step", "train_loss", "val_loss", "lr"])
    if write_header:
        writer.writeheader()
        csv_file.flush()

    to_skip = start_step * accum
    train_iter = iter(train_loader)
    skipped = 0
    while skipped < to_skip:
        try:
            next(train_iter)
            skipped += 1
        except StopIteration:
            train_iter = iter(train_loader)

    step = start_step
    if step == 0 and eval_every > 0:
        v0 = evaluate(model, val_loader, vocab_size, unk_id, args.max_val_batches, device)
        writer.writerow({"step": 0, "train_loss": "", "val_loss": f"{v0:.6f}", "lr": f"{opt.param_groups[0]['lr']:.8f}"})
        csv_file.flush()
        print(f"step 0 val={v0:.4f}")

    while step < max_steps:
        opt.zero_grad(set_to_none=True)
        acc_loss = 0.0
        for _ in range(accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            input_ids = torch.where(input_ids >= vocab_size, torch.tensor(unk_id, device=input_ids.device), input_ids)
            labels = torch.where(
                (labels != -100) & (labels >= vocab_size),
                torch.tensor(unk_id, dtype=labels.dtype, device=labels.device),
                labels,
            )
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )
            (loss / accum).backward()
            acc_loss += float(loss.item())
        acc_loss /= 1.0
        avg_loss = acc_loss / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        sched.step()
        step += 1
        lr_now = float(opt.param_groups[0]["lr"])

        do_eval = (eval_every > 0 and (step % eval_every == 0 or step == max_steps))
        val_loss = ""
        vnum = None
        if do_eval:
            vnum = evaluate(model, val_loader, vocab_size, unk_id, args.max_val_batches, device)
            val_loss = f"{vnum:.6f}"
            if vnum < best_val:
                best_val = vnum
                best_path = ckpt_dir / f"best_{preset}.pt"
                _rs = get_rng_states()
                _payload = {
                    "step": step, "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(), "scheduler_state": sched.state_dict(),
                    "best_val": best_val, "train_loss": avg_loss, "val_loss": vnum,
                    "rng_python": _rs["python"],
                    "rng_numpy": _rs["numpy"],
                    "rng_torch": _rs["torch"],
                    "preset": preset, "seq_len": seq_len,
                    "effective_batch": effective, "micro_batch": micro,
                }
                if "cuda" in _rs:
                    _payload["rng_cuda"] = _rs["cuda"]
                save_checkpoint(best_path, _payload)
        writer.writerow({
            "step": step, "train_loss": f"{avg_loss:.6f}",
            "val_loss": val_loss, "lr": f"{lr_now:.8f}",
        })
        csv_file.flush()

        if step % args.log_every == 0 or step == max_steps or do_eval:
            msg = f"step {step}/{max_steps} loss={avg_loss:.4f} lr={lr_now:.2e}"
            if vnum is not None:
                msg += f" val={vnum:.4f} best={best_val:.4f}"
            print(msg, flush=True)

        if save_every > 0 and (step % save_every == 0 or step == max_steps):
            ckpt_path = ckpt_dir / f"ckpt_{preset}_step{step}.pt"
            rs = get_rng_states()
            base_payload = {
                "step": step, "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(), "scheduler_state": sched.state_dict(),
                "best_val": best_val, "train_loss": avg_loss,
                "val_loss": vnum if vnum is not None else float("nan"),
                "rng_python": rs["python"], "rng_numpy": rs["numpy"], "rng_torch": rs["torch"],
                "preset": preset, "seq_len": seq_len,
                "effective_batch": effective, "micro_batch": micro,
            }
            if "cuda" in rs:
                base_payload["rng_cuda"] = rs["cuda"]
            save_checkpoint(ckpt_path, base_payload)
            save_checkpoint(ckpt_dir / f"last_{preset}.pt", dict(base_payload))
            print(f"saved {ckpt_path}")

    csv_file.close()
    print(f"done preset={preset} steps={max_steps} best_val={best_val:.4f} log={csv_path}")


if __name__ == "__main__":
    main()
