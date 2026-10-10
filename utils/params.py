from __future__ import annotations

from typing import Dict, Tuple


def _unique_params(model):
    seen = set()
    uniq = []
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            uniq.append(p)
    return uniq


def count_parameters(model, unique: bool = True) -> Tuple[int, int]:
    if unique:
        params = _unique_params(model)
        total = sum(p.numel() for p in params)
        trainable = sum(p.numel() for p in params if p.requires_grad)
        return total, trainable
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_parameters_raw(model) -> Tuple[int, int]:
    return count_parameters(model, unique=False)


def per_block_counts(model) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for name, module in model.named_children():
        counts[name] = sum(p.numel() for p in module.parameters())
    counts["<other>"] = sum(
        p.numel() for n, p in model.named_parameters() if "." not in n
    )
    if counts["<other>"] == 0:
        del counts["<other>"]
    return counts


def print_per_block_params(model) -> Dict[str, int]:
    total, trainable = count_parameters(model)
    blocks = per_block_counts(model)
    print(f"total={total:,} trainable={trainable:,}")
    for name, n in blocks.items():
        pct = (100.0 * n / total) if total else 0.0
        print(f"  {name}: {n:,} ({pct:.2f}%)")
    return blocks


def param_breakdown(model) -> Dict[str, int]:
    out: Dict[str, int] = {}
    seen = set()

    def _add(key: str, p) -> None:
        if id(p) in seen:
            out.setdefault(key + " (shared, counted once)", 0)
            return
        seen.add(id(p))
        out[key] = out.get(key, 0) + p.numel()

    for name, p in model.named_parameters():
        top = name.split(".")[0]
        if top in ("tok_emb", "token_embedding", "wte", "embed"):
            _add("embed", p)
        elif top in ("lm_head", "head", "logits"):
            _add("lm_head", p)
        elif top in ("blocks", "layers", "h", "transformer"):
            _add("blocks", p)
        elif top in ("final_norm", "ln_f", "norm_f", "final_ln"):
            _add("final_norm", p)
        elif top in ("pos_emb", "wpe", "pos_embedding"):
            _add("pos_emb", p)
        else:
            _add(top, p)
    return out


def print_param_table(model, name: str = "model") -> Dict[str, int]:
    total, trainable = count_parameters(model)
    blocks = param_breakdown(model)
    try:
        tok = model.tok_emb.weight  # type: ignore[attr-defined]
        head = model.lm_head.weight  # type: ignore[attr-defined]
        if tok is head and "lm_head" not in blocks:
            blocks["lm_head (tied)"] = 0
    except AttributeError:
        pass
    print(f"[{name}] total={total:,} trainable={trainable:,}")
    for k in sorted(blocks):
        n = blocks[k]
        pct = (100.0 * n / total) if total else 0.0
        print(f"  {k:12s} {n:>12,} ({pct:5.2f}%)")
    return blocks
