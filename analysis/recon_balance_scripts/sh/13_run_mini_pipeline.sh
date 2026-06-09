#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="${1:-$ROOT_DIR/00_config_example.yaml}"
SPLIT_ROOT="${2:-/home/shukang/datasets/split_files}"
OUT_YAML="${3:-/home/shukang/datasets/VisDrone_ReCon_Data_mini.yaml}"

bash "$SCRIPT_DIR/08_build_dataset_mini.sh" "$CONFIG"
bash "$SCRIPT_DIR/10_write_data_yaml.sh" mini "$SPLIT_ROOT" "$OUT_YAML"
bash "$SCRIPT_DIR/11_train_mini.sh"
