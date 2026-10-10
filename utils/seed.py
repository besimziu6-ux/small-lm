from __future__ import annotations

import os
import random


def set_all_seeds(seed: int = 42, threads: int = 4, deterministic: bool = True) -> int:
    import sys

    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError as e:
        print(f"warning: interop threads already set ({e}), keeping existing", file=sys.stderr)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError as e:
            print(f"warning: deterministic algorithms unavailable ({e}), continuing non-deterministically", file=sys.stderr)
    return seed


def wants_deterministic(device: str) -> bool:
    import sys

    if str(device).lower() == "cuda" and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        print("warning: CUBLAS_WORKSPACE_CONFIG not set, disabling deterministic algorithms on CUDA "
              "(export CUBLAS_WORKSPACE_CONFIG=:4096:8 to re-enable)", file=sys.stderr)
        return False
    return True
