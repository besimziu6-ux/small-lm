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
        import os

        import torch

        torch.set_num_threads(self.threads)
        try:
            torch.set_num_interop_threads(self.threads)
        except RuntimeError as e:
            print(f"warning: interop threads already set ({e}), keeping existing", file=sys.stderr)
        torch.set_default_dtype(getattr(torch, self.dtype))
        torch.backends.cudnn.benchmark = self.cudnn_benchmark
        torch.backends.cudnn.deterministic = True
        want_det = bool(self.use_deterministic_algorithms)
        if want_det and str(self.device).lower() == "cuda" and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            print("warning: CUBLAS_WORKSPACE_CONFIG not set, disabling deterministic algorithms on CUDA "
                  "(export CUBLAS_WORKSPACE_CONFIG=:4096:8 to re-enable)", file=sys.stderr)
            want_det = False
        if want_det:
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
