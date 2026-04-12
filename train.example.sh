#!/bin/bash

# 项目级配置，避免和其他项目冲突
export YOLO_CONFIG_DIR="$(cd "$(dirname "$0")" && pwd)/.config"
# 确保配置目录存在
mkdir -p "$YOLO_CONFIG_DIR/Ultralytics"

# 数据集和模型选择
DATA="VisDrone"
MODEL="yolo26m"

# === Resume 续训模式 ===
# 取消下面两行注释即可续训，注释掉下方新训练部分
# RESUME="runs/detect/visdrone/yolo26m-20260411-221639/weights/last.pt"
# LOG_FILE="logs/${DATA}-${MODEL}-resume-$(date +%Y%m%d-%H%M%S).log"
# python train.py --resume "$RESUME" > "$LOG_FILE" 2>&1 &

# === 新训练模式 ===
BATCH=4     # 总 batch size，多卡时自动平分到每张卡
DEVICE="0,1" # "cpu" or GPU ID，"0,1" 表示使用两个GPU，batch平分到2张卡上
LOG_FILE="logs/${DATA}-${MODEL}-$(date +%Y%m%d-%H%M%S).log"
python train.py --model "$MODEL" --data "$DATA" --batch $BATCH --device "$DEVICE" > "$LOG_FILE" 2>&1 &

echo "Training started, log: $LOG_FILE"
# ps aux |grep 'python train.py' 手动确认是否要kill掉之前的训练进程
