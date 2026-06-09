#!/bin/bash
set -euo pipefail

YOLO_ROOT="/home/shukang/project/YOLO26"
PYTHON_BIN="/home/shukang/miniconda3/envs/yolo26/bin/python"
LOG_DIR="$YOLO_ROOT/log"

CLASS_ID="${1:-6}"
MODEL="${MODEL:-yolo26n-nop5}"
BATCH="${BATCH:-4}"
DEVICE="${DEVICE:-0,1}"
EPOCHS="${EPOCHS:-120}"
PATIENCE="${PATIENCE:-10}"
IMGSZ="${IMGSZ:-1280}"
MULTI_SCALE="${MULTI_SCALE:-0}"
LR0="${LR0:-0.01}"

mkdir -p "$LOG_DIR"

DATA="/home/shukang/project/recon_workbench/split_files_tail5/VisDrone_Tail5Chip_cls${CLASS_ID}.yaml"
TRAIN_SCRIPT="$YOLO_ROOT/train_tail5.py"

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/VisDrone_Tail5_cls${CLASS_ID}-${MODEL}-b${BATCH}-s${IMGSZ}-ms${MULTI_SCALE}-${STAMP}.log"

{
  echo "============================================================"
  echo "[START] $(date '+%F %T')"
  echo "[CLASS] $CLASS_ID"
  echo "[MODEL] $MODEL"
  echo "[DATA]  $DATA"
  echo "[PY]    $PYTHON_BIN"
  echo "[LOG]   $LOG_FILE"
  echo "[CFG]   batch=$BATCH device=$DEVICE epochs=$EPOCHS patience=$PATIENCE imgsz=$IMGSZ ms=$MULTI_SCALE lr0=$LR0"
  echo "============================================================"
} | tee -a "$LOG_FILE"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[FATAL] python not found: $PYTHON_BIN" | tee -a "$LOG_FILE"
  exit 1
fi
if [ ! -f "$TRAIN_SCRIPT" ]; then
  echo "[FATAL] train script not found: $TRAIN_SCRIPT" | tee -a "$LOG_FILE"
  exit 1
fi
if [ ! -f "$DATA" ]; then
  echo "[FATAL] data yaml not found: $DATA" | tee -a "$LOG_FILE"
  exit 1
fi

cd "$YOLO_ROOT"
export YOLO_CONFIG_DIR="$YOLO_ROOT/.config"
mkdir -p "$YOLO_CONFIG_DIR/Ultralytics"

CMD="cd '$YOLO_ROOT' && PYTHONUNBUFFERED=1 stdbuf -oL -eL '$PYTHON_BIN' -u train_tail5.py --model '$MODEL' --data '$DATA' --batch '$BATCH' --device '$DEVICE' --epochs '$EPOCHS' --patience '$PATIENCE' --imgsz '$IMGSZ' --multi_scale '$MULTI_SCALE' --lr0 '$LR0'"

echo "[CMD] $CMD" | tee -a "$LOG_FILE"
nohup bash -lc "$CMD" >> "$LOG_FILE" 2>&1 &
PID=$!
echo "[PID] $PID" | tee -a "$LOG_FILE"
echo "[TAIL] tail -f $LOG_FILE" | tee -a "$LOG_FILE"
echo "Training started in background."
