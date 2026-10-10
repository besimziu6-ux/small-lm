#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

echo "== 1/6 param count =="
"$PY" -m pytest tests/test_param_count.py -q
echo "param count check green"

echo "== 2/6 shapes =="
"$PY" -m pytest tests/test_shapes.py -q
echo "shape check green"

echo "== 3/6 determinism (seed) =="
"$PY" -m pytest tests/test_determinism.py -q
echo "seed check green"

echo "== 4/6 loss down =="
"$PY" -m pytest tests/test_loss_down.py -q
echo "loss check green"

echo "== 5/6 tokenizer =="
"$PY" -m pytest tests/test_tokenizer.py -q
echo "tokenizer check green"

echo "== 6/6 generate =="
"$PY" -m pytest tests/test_generate.py -q
echo "generation check green"

echo "ALL GREEN: param count, seed, shape, loss, tokenizer, generation"
