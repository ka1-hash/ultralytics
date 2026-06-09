#!/usr/bin/env bash
set -euo pipefail

# Edit these variables for your environment.
YOLO_ROOT="${YOLO_ROOT:-/home/shukang/yolo26_dev}"
DATA_YAML="${DATA_YAML:-/home/shukang/datasets/VisDrone_ReCon_Data_full.yaml}"
MODEL_CFG="${MODEL_CFG:-ultralytics/cfg/models/26/yolo26-nop5.yaml}"
PRETRAINED="${PRETRAINED:-weights/yolo26n.pt}"
PROJECT="${PROJECT:-runs/recon_balance}"
NAME="${NAME:-recon_full_yolo26n_nop5_s1280}"
DEVICE="${DEVICE:-0}"
IMGSZ="${IMGSZ:-1280}"
BATCH="${BATCH:-4}"
EPOCHS="${EPOCHS:-120}"

cd "$YOLO_ROOT"
python train.py \
  --model "$MODEL_CFG" \
  --data "$DATA_YAML" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --epochs "$EPOCHS" \
  --device "$DEVICE" \
  --optimizer MuSGD \
  --lr0 0.01 \
  --mosaic 1.0 \
  --close_mosaic 10 \
  --multi_scale 0 \
  --copy_paste 0.0 \
  --mixup 0.0 \
  --cutmix 0.0 \
  --project "$PROJECT" \
  --name "$NAME" \
  --pretrained "$PRETRAINED"
