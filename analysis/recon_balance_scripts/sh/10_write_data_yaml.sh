#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash 10_write_data_yaml.sh mini /home/shukang/datasets/split_files /home/shukang/datasets/VisDrone_ReCon_Data_mini.yaml
#   bash 10_write_data_yaml.sh full /home/shukang/datasets/split_files /home/shukang/datasets/VisDrone_ReCon_Data_full.yaml

STAGE="${1:-mini}"
SPLIT_ROOT="${2:-/home/shukang/datasets/split_files}"
OUT_YAML="${3:-/home/shukang/datasets/VisDrone_ReCon_Data_${STAGE}.yaml}"

cat > "$OUT_YAML" <<YAML
path: /home/shukang/datasets
train: ${SPLIT_ROOT}/train_union_${STAGE}.txt
val: ${SPLIT_ROOT}/val_raw.txt

names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
YAML

echo "Wrote $OUT_YAML"
