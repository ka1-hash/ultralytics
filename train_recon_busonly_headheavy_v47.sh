#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export YOLO_CONFIG_DIR="$(pwd)/.config"
mkdir -p "$YOLO_CONFIG_DIR/Ultralytics" logs

DATA="${DATA:-/home/shukang/project/recon_workbench/split_files_busfirst_v47/VisDrone_AllTail.yaml}"
BASELINE_WEIGHTS="${BASELINE_WEIGHTS:-/home/shukang/project/YOLO26/best.pt}"
MODEL="${MODEL:-yolo26n-nop5}"
DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-4}"
IMGSZ="${IMGSZ:-1280}"
MULTI_SCALE="${MULTI_SCALE:-0}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-5}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-15}"
FREEZE_LAYERS="${FREEZE_LAYERS:-10}"
LR_STAGE1="${LR_STAGE1:-0.0025}"
LR_STAGE2="${LR_STAGE2:-0.0015}"
PATIENCE="${PATIENCE:-10}"

STAMP=$(date +%Y%m%d-%H%M%S)
LOG1="logs/busfirst-stage1-${STAMP}.log"
LOG2="logs/busfirst-stage2-${STAMP}.log"

echo "[Stage1] head-heavy warmup from baseline: ${BASELINE_WEIGHTS}" | tee "$LOG1"
python train_busfirst.py   --weights_init "$BASELINE_WEIGHTS"   --data "$DATA"   --batch "$BATCH"   --device "$DEVICE"   --epochs "$STAGE1_EPOCHS"   --patience "$PATIENCE"   --imgsz "$IMGSZ"   --multi_scale "$MULTI_SCALE"   --freeze "$FREEZE_LAYERS"   --lr0 "$LR_STAGE1"   --name_prefix busfirst-stage1   > "$LOG1" 2>&1

STAGE1_BEST=$(find . -path "*busfirst-stage1*weights/best.pt" | sort | tail -n 1)
if [ -z "$STAGE1_BEST" ]; then
  echo "Stage1 best.pt not found" >&2
  exit 1
fi

echo "[Stage2] light full-network finetune from ${STAGE1_BEST}" | tee "$LOG2"
python train.py   --weights_init "$STAGE1_BEST"   --data "$DATA"   --batch "$BATCH"   --device "$DEVICE"   --epochs "$STAGE2_EPOCHS"   --patience "$PATIENCE"   --imgsz "$IMGSZ"   --multi_scale "$MULTI_SCALE"   --freeze 0   --lr0 "$LR_STAGE2"   --name_prefix busfirst-stage2   > "$LOG2" 2>&1

echo "Done. Logs: $LOG1 $LOG2"
