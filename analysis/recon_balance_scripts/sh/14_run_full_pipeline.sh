#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="${1:-$ROOT_DIR/00_config_example.yaml}"
SPLIT_ROOT="${2:-/home/shukang/datasets/split_files}"
OUT_YAML="${3:-/home/shukang/datasets/VisDrone_ReCon_Data_full.yaml}"

bash "$SCRIPT_DIR/09_build_dataset_full.sh" "$CONFIG"
bash "$SCRIPT_DIR/10_write_data_yaml.sh" full "$SPLIT_ROOT" "$OUT_YAML"
bash "$SCRIPT_DIR/12_train_full.sh"
