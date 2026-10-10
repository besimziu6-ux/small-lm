# small-lm

Minimal CPU training base. Python 3.12.3, float32, 4 threads, deterministic.

## Setup

Requires Python 3.12.3:

```bash
python3 --version  # expect Python 3.12.3
pip install -r training/small-lm/requirements.txt
```

The requirements file pulls torch from the CPU index
(`https://download.pytorch.org/whl/cpu`) with pinned
torch, tokenizers, numpy, tqdm, pytest.

## Smoke run

From repo root (`/home/besi`):

```bash
cd training/small-lm
python -c "from config import Config; print(Config())"
python -c "from utils.seed import set_all_seeds; import torch; set_all_seeds(42); print(torch.randn(3, 3))"
python -c "from utils.seed import set_all_seeds; import torch; set_all_seeds(42); print(torch.randn(3, 3))"
```

The last two commands print identical tensors (seed 42 is deterministic).

Param breakdown:

```bash
python -c "import torch; from config import Config; from utils.seed import set_all_seeds; from utils.params import print_per_block_params; set_all_seeds(Config().seed); m=torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(32,4,64,batch_first=True),2); print_per_block_params(m)"
```

## Defaults

- `seed=42`, `dtype=float32`, `device=cpu`
- `threads=4`, `cudnn_benchmark=False`, `use_deterministic_algorithms=True`
- Keeps RAM under 7.6GB. No CUDA needed.
