#!/usr/bin/env bash
set -euo pipefail

DATA_PATH=${1:-/Users/harryvoorhis/Downloads/org-redox-dataset-main}
OUT_DIR=${2:-results/default_run}

redox-train \
  --data-path "$DATA_PATH" \
  --output-dir "$OUT_DIR" \
  --smiles-col smiles \
  --target-col deltaE_V \
  --fingerprint-bits 2048 \
  --fingerprint-radius 2 \
  --test-size 0.2 \
  --random-state 42 \
  --n-jobs -1
