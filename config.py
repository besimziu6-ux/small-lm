from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys


def resolve_device(name: str = "cpu") -> str:
    import torch

    n = str(name).lower().strip()
    if n == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if n == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested (--device cuda) but torch.cuda.is_available() is False")
        return "cuda"
    return "cpu"


@dataclass
class Config:
    seed: int = 42
    dtype: str = "float32"
    device: str = "cpu"
    threads: int = 4
    use_deterministic_algorithms: bool = True
    cudnn_benchmark: bool = False
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent
    )
    data_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "data"
    )
    checkpoint_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "checkpoints"
    )
    log_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "logs"
    )

    def apply_torch_settings(self) -> None:
        import torch

        torch.set_num_threads(self.threads)
        torch.set_num_interop_threads(self.threads)
        torch.set_default_dtype(getattr(torch, self.dtype))
        torch.backends.cudnn.benchmark = self.cudnn_benchmark
        torch.backends.cudnn.deterministic = True
        if self.use_deterministic_algorithms:
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError as e:
                print(f"warning: deterministic algorithms unavailable ({e}), continuing non-deterministically", file=sys.stderr)

    def torch_dtype(self):
        import torch

        return getattr(torch, self.dtype)

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
