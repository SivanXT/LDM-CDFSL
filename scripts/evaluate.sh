#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the directory containing the target datasets.}"
: "${CHECKPOINT:?Set CHECKPOINT to a checkpoint produced by this code.}"

DATASET="${DATASET:-isic}"
SHOT="${SHOT:-1}"
DEVICE="${DEVICE:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-outdir/predictions}"

command=(
  python prompt_test.py
  --data-root "${DATA_ROOT}"
  --dataset "${DATASET}"
  --shot "${SHOT}"
  --backbone resnet50
  --checkpoint "${CHECKPOINT}"
  --refine
  --device "${DEVICE}"
  --output-dir "${OUTPUT_DIR}"
)
if [[ -n "${BACKBONE_WEIGHTS:-}" ]]; then
  command+=(--backbone-weights "${BACKBONE_WEIGHTS}")
fi
"${command[@]}"
