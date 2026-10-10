from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(p: str | Path) -> Path:
    q = Path(p)
    if q.is_absolute():
        return q
    cwd_q = Path.cwd() / q
    if Path.cwd().name == "small-lm":
        return cwd_q
    return _PROJECT_ROOT / q
