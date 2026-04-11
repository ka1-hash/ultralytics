#!/bin/bash

# 项目级配置，避免和其他项目冲突
export YOLO_CONFIG_DIR="$(cd "$(dirname "$0")" && pwd)/.config"
# 确保配置目录存在
mkdir -p "$YOLO_CONFIG_DIR/Ultralytics"

# 生成带时间戳的日志文件名
LOG_FILE="logs/vis-26n-$(date +%Y%m%d-%H%M%S).log"
CUDA_VISIBLE_DEVICES=3 python train.py > "$LOG_FILE" 2>&1 &

echo "Training started, log: $LOG_FILE"
# ps aux |grep 'python train.py' 手动确认是否要kill掉之前的训练进程
