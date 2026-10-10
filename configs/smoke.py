from __future__ import annotations

PRESET = "smoke"

vocab_size = 8000
d_model = 256
n_layer = 6
n_head = 4
d_ff = 512
max_seq_len = 256
pos = "rope"
norm = "rmsnorm"
mlp = "gelu"
tie_word_embeddings = True
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
