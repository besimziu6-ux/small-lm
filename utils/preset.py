from __future__ import annotations

import sys
from pathlib import Path


def normalize_preset(name: str) -> str:
    n = str(name).lower().strip()
    if n in ("smoke",):
        return "smoke"
    if n in ("base", "base_65m", "base-65m"):
        return "base"
    raise ValueError(f"unknown preset {name!r}")


def infer_preset_and_seq(
    ckpt: dict,
    fallback_preset: str | None = None,
    fallback_seq: int | None = None,
    ckpt_path: str | Path | None = None,
    default_seq: int = 128,
) -> tuple[str, int]:
    if fallback_preset is None:
        raw = ckpt.get("preset", "smoke") if isinstance(ckpt, dict) else "smoke"
    else:
        raw = fallback_preset
    try:
        preset = normalize_preset(raw)
    except (ValueError, AttributeError):
        preset = "smoke"
    seq = fallback_seq
    if seq is None:
        try:
            raw_seq = ckpt.get("seq_len", default_seq) if isinstance(ckpt, dict) else default_seq
            seq = int(raw_seq)
        except (ValueError, TypeError) as e:
            print(
                f"seq_len parse failed ({ckpt_path if ckpt_path is not None else 'unknown checkpoint'}: {e}), using default {default_seq}",
                file=sys.stderr,
            )
            seq = default_seq
    return preset, int(seq)
