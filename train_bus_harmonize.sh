#!/bin/bash
# 训练 bus harmonize 数据集

cd "$(dirname "$0")"
export YOLO_CONFIG_DIR="$(pwd)/.config"
mkdir -p "$YOLO_CONFIG_DIR/Ultralytics"
mkdir -p "logs"

DATA="VisDrone_BusHarmonize"
MODEL="yolo26n-nop5"

BATCH=4
DEVICE="0,1"
EPOCHS=120
PATIENCE=10
IMGSZ="1280"
MULTI_SCALE=0

LOG_FILE="logs/${DATA}-${MODEL}-b${BATCH}-s${IMGSZ}-ms${MULTI_SCALE}-$(date +%Y%m%d-%H%M%S).log"

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
