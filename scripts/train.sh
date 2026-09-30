#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the directory containing VOC2012 and the target datasets.}"
: "${BACKBONE_WEIGHTS:?Set BACKBONE_WEIGHTS to a compatible ImageNet-pretrained ResNet state dict.}"

DATASET="${DATASET:-isic}"
SHOT="${SHOT:-1}"
DEVICE="${DEVICE:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-outdir/models}"

python prompt_train.py \
  --data-root "${DATA_ROOT}" \
  --dataset "${DATASET}" \
  --shot "${SHOT}" \
  --backbone resnet50 \
  --backbone-weights "${BACKBONE_WEIGHTS}" \
  --refine \
  --device "${DEVICE}" \
  --output-dir "${OUTPUT_DIR}"
