#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="${1:-$ROOT_DIR/00_config_example.yaml}"

python "$ROOT_DIR/01_build_instance_bank.py" --config "$CONFIG"
python "$ROOT_DIR/02_build_position_candidates.py" --config "$CONFIG"
python "$ROOT_DIR/03_make_recon_tasks.py" --config "$CONFIG" --stage mini
python "$ROOT_DIR/04_run_recon_batch.py" --config "$CONFIG" --stage mini
python "$ROOT_DIR/05_qc_recon_outputs.py" --config "$CONFIG" --stage mini
python "$ROOT_DIR/06_assemble_balance_dataset.py" --config "$CONFIG" --stage mini
python "$ROOT_DIR/07_make_split_lists.py" --config "$CONFIG" --stage mini
