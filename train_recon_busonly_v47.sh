#!/bin/bash
# Bus-First Progressive Hybrid ReCon 训练（单阶段）
cd "$(dirname "$0")"
export YOLO_CONFIG_DIR="$(pwd)/.config"
mkdir -p "$YOLO_CONFIG_DIR/Ultralytics" logs

DATA="/home/shukang/project/recon_workbench/split_files_busfirst_v47/VisDrone_AllTail.yaml"
MODEL="${MODEL:-yolo26n-nop5}"
BATCH="${BATCH:-4}"
DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-120}"
PATIENCE="${PATIENCE:-10}"
IMGSZ="${IMGSZ:-1280}"
MULTI_SCALE="${MULTI_SCALE:-0}"
LR0="${LR0:-0.01}"

LOG_FILE="logs/VisDrone_BusFirst-${MODEL}-b${BATCH}-s${IMGSZ}-ms${MULTI_SCALE}-$(date +%Y%m%d-%H%M%S).log"
python train.py   --model "$MODEL"   --data "$DATA"   --batch "$BATCH"   --device "$DEVICE"   --epochs "$EPOCHS"   --patience "$PATIENCE"   --imgsz "$IMGSZ"   --multi_scale "$MULTI_SCALE"   --lr0 "$LR0"   --name_prefix busfirst   > "$LOG_FILE" 2>&1 &

echo "Training started, log: $LOG_FILE"
echo "PID: $!"
