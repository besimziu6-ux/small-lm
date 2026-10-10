from __future__ import annotations

PRESET = "gpt2_124m"

vocab_size = 32000
d_model = 768
n_layer = 12
n_head = 12
d_ff = 3072
max_seq_len = 512
pos = "rope"
norm = "rmsnorm"
mlp = "gelu"
tie_word_embeddings = False
dropout = 0.0


def to_dict() -> dict:
    return {
        "preset": PRESET,
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_layer": n_layer,
        "n_head": n_head,
        "d_ff": d_ff,
        "max_seq_len": max_seq_len,
        "pos": pos,
        "norm": norm,
        "mlp": mlp,
        "tie_word_embeddings": tie_word_embeddings,
        "dropout": dropout,
    }
