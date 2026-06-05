#!/bin/bash
# Bus-First ReCon + Chip 训练: yolo26n-nop5
cd "$(dirname "$0")"
export YOLO_CONFIG_DIR="$(pwd)/.config"
mkdir -p "$YOLO_CONFIG_DIR/Ultralytics"
mkdir -p "logs"

DATA="/home/shukang/project/recon_workbench/split_files_busfirst_v47_chip/VisDrone_InPlaceChip_cls8.yaml"
MODEL="yolo26n-nop5"

BATCH=4
DEVICE="0"
EPOCHS=120
PATIENCE=10
IMGSZ="1280"
MULTI_SCALE=0

DATA_TAG="VisDrone_BusFirstChip"
LOG_FILE="logs/${DATA_TAG}-${MODEL}-b${BATCH}-s${IMGSZ}-ms${MULTI_SCALE}-$(date +%Y%m%d-%H%M%S).log"

python train.py \
  --model "$MODEL" \
  --data "$DATA" \
  --batch $BATCH \
  --device "$DEVICE" \
  --epochs $EPOCHS \
  --patience $PATIENCE \
  --imgsz "$IMGSZ" \
  --multi_scale $MULTI_SCALE \
  > "$LOG_FILE" 2>&1 &

echo "Training started, log: $LOG_FILE"
echo "PID: $!"