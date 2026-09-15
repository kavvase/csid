#!/usr/bin/env sh
set -eu
DATA_ROOT=${1:-data/instacart}
OUTPUT=${2:-outputs/item-universe-expansion}
uv run csid universe \
  --data-root "$DATA_ROOT" \
  --output "$OUTPUT" \
  --n-users 30000 \
  --focus-n 20 \
  --universe-sizes 20,40,80,all \
  --triple-min-count 30
