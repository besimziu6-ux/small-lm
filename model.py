from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch
import torch.nn.functional as F

from config import Config, resolve_device
from utils.seed import set_all_seeds, wants_deterministic
from utils.params import count_parameters, print_param_table


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 512
    n_layer: int = 8
    n_head: int = 8
    d_ff: int = 2048
    max_seq_len: int = 512
    pos: str = "rope"
    norm: str = "rmsnorm"
    mlp: str = "gelu"
    tie_word_embeddings: bool = False
    dropout: float = 0.0


def _load_preset_dict(preset: str) -> dict:
    key = preset.lower().strip()
    if key in ("smoke",):
        from configs.smoke import to_dict

        return to_dict()
    if key in ("base", "base_65m", "base-65m"):
        from configs.base_65m import to_dict

        return to_dict()
    raise ValueError(f"unknown preset {preset!r}; expected 'smoke' or 'base'")


def preset_to_config(preset: str) -> ModelConfig:
    d = _load_preset_dict(preset)
    return ModelConfig(
        vocab_size=int(d["vocab_size"]),
        d_model=int(d["d_model"]),
        n_layer=int(d["n_layer"]),
        n_head=int(d["n_head"]),
        d_ff=int(d["d_ff"]),
        max_seq_len=int(d.get("max_seq_len", 512)),
        pos=str(d.get("pos", "rope")),
        norm=str(d.get("norm", "rmsnorm")),
        mlp=str(d.get("mlp", "gelu")),
        tie_word_embeddings=bool(d.get("tie_word_embeddings", False)),
        dropout=float(d.get("dropout", 0.0)),
    )


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.eps)
        return (self.weight * xf).to(dt)


def _build_norm(kind: str, dim: int) -> torch.nn.Module:
    if kind.lower() in ("rmsnorm", "rms", "rms_norm"):
        return RMSNorm(dim)
    return torch.nn.LayerNorm(dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def _build_rope_cache(seq_len: int, head_dim: int, device, dtype):
    assert head_dim % 2 == 0
    inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _apply_rope(q, k, cos, sin):
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class CausalSelfAttention(torch.nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.q_proj = torch.nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = torch.nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = torch.nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = torch.nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = torch.nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cos=None, sin=None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if cos is not None and sin is not None:
            c = cos[:T].to(q.dtype).unsqueeze(0).unsqueeze(0)
            s = sin[:T].to(q.dtype).unsqueeze(0).unsqueeze(0)
            q, k = _apply_rope(q, k, c, s)
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
            att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att.float(), dim=-1).to(x.dtype)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.drop(self.o_proj(y))


class MLP(torch.nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.kind = cfg.mlp.lower()
        if self.kind in ("swiglu", "swi_glu", "swi-glu"):
            self.gate_proj = torch.nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.up_proj = torch.nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.down_proj = torch.nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        else:
            self.fc1 = torch.nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.fc2 = torch.nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
            self.act = torch.nn.GELU()
        self.drop = torch.nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind in ("swiglu", "swi_glu", "swi-glu"):
            return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Block(torch.nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = _build_norm(cfg.norm, cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = _build_norm(cfg.norm, cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, cos=None, sin=None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class SmallLM(torch.nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = torch.nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_kind = cfg.pos.lower()
        if self.pos_kind in ("learned", "absolute", "embedding"):
            self.pos_emb = torch.nn.Embedding(cfg.max_seq_len, cfg.d_model)
        else:
            self.pos_emb = None
        self.blocks = torch.nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = _build_norm(cfg.norm, cfg.d_model)
        self.lm_head = torch.nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        rope_hd = cfg.d_model // cfg.n_head
        self._rope_hd = rope_hd
        self.register_buffer("rope_cos", torch.zeros(1), persistent=False)
        self.register_buffer("rope_sin", torch.zeros(1), persistent=False)
        self._init_rope(cfg.max_seq_len, rope_hd)

    def _init_rope(self, seq_len: int, head_dim: int) -> None:
        if self.pos_kind in ("learned", "absolute", "embedding"):
            return
        cos, sin = _build_rope_cache(seq_len, head_dim, device="cpu", dtype=torch.float32)
        self.rope_cos = cos
        self.rope_sin = sin

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}; use --seq-len <= {self.cfg.max_seq_len}")
        x = self.tok_emb(input_ids)
        if self.pos_emb is not None:
            pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
            x = x + self.pos_emb(pos)
            cos = sin = None
        else:
            if self.rope_cos.numel() <= 1 or self.rope_cos.shape[0] < T:
                cos, sin = _build_rope_cache(T, self._rope_hd, device=x.device, dtype=torch.float32)
                cos = cos.to(x.device)
                sin = sin.to(x.device)
            else:
                cos = self.rope_cos.to(x.device)
                sin = self.rope_sin.to(x.device)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.final_norm(x)
        return self.lm_head(x)


def _init_weights(model: SmallLM) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "weight" in name and p.dim() >= 2:
                if model.cfg.tie_word_embeddings and name == "lm_head.weight":
                    continue
                torch.nn.init.normal_(p, mean=0.0, std=0.02)
            elif "weight" in name and p.dim() == 1:
                torch.nn.init.ones_(p)
            elif "bias" in name:
                torch.nn.init.zeros_(p)
        if model.cfg.tie_word_embeddings:
            model.lm_head.weight = model.tok_emb.weight


def build_model(preset: str = "base", seed: int = 42, base_cfg: Config | None = None) -> SmallLM:
    gc = base_cfg or Config()
    try:
        gc.apply_torch_settings()
    except RuntimeError as e:
        print(f"warning: torch settings failed ({e}), using defaults", file=sys.stderr)
        torch.set_num_threads(gc.threads)
    try:
        set_all_seeds(seed if seed is not None else gc.seed, threads=gc.threads,
                        deterministic=wants_deterministic(getattr(gc, "device", "cpu")))
    except RuntimeError as e:
        print(f"warning: deterministic seeding failed ({e}), continuing non-deterministically", file=sys.stderr)
        torch.manual_seed(seed if seed is not None else gc.seed)
    try:
        _dev = resolve_device(getattr(gc, "device", "cpu"))
    except RuntimeError:
        raise
    except Exception:
        _dev = "cpu"
    if _dev == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed if seed is not None else gc.seed)
        except Exception as e:
            print(f"warning: cuda seeding failed ({e})", file=sys.stderr)
    torch.set_default_dtype(torch.float32)

    cfg = preset_to_config(preset)
    model = SmallLM(cfg)
    _init_weights(model)
    if cfg.tie_word_embeddings:
        model.lm_head.weight = model.tok_emb.weight
    model = model.to(torch.float32).eval()
    try:
        model.to(_dev)
    except Exception as e:
        print(f"warning: move to device {_dev} failed ({e}), staying on cpu", file=sys.stderr)

    total, _ = count_parameters(model)
    if preset.lower().strip() in ("base", "base_65m", "base-65m"):
        assert 50_000_000 <= total <= 100_000_000, f"base total {total:,} outside 50M-100M"
    return model


def count_model_params(model: SmallLM) -> Tuple[int, int]:
    return count_parameters(model)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="base")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    m = build_model(args.preset, seed=args.seed)
    print_param_table(m, name=args.preset)
