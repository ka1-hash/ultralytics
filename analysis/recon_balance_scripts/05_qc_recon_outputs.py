from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from common import artifact_path, load_csv, load_yaml, parse_stage, save_csv


def qc_placeholder(class_id: int, s_pos: float, q_src: float, version: int) -> Dict[str, float]:
    # 占位版分数：真实项目中请替换为 GDINO / SAM / YOLO26 推理结果
    base = 0.02 * (version % 3)
    q_gdino = min(0.95, 0.72 + base + (0.03 if class_id == 8 else 0.0))
    q_sam = min(0.95, 0.75 + base + (0.02 if class_id == 7 else 0.0))
    q_yolo = min(0.95, 0.68 + base + (0.04 if class_id == 8 else 0.0))
    q_div = 0.75 - 0.05 * (version % 2)
    q_final = 0.20 * q_src + 0.20 * s_pos + 0.20 * q_gdino + 0.20 * q_sam + 0.15 * q_yolo + 0.05 * q_div
    return {
        "Q_gdino": round(q_gdino, 4),
        "Q_sam": round(q_sam, 4),
        "Q_yolo": round(q_yolo, 4),
        "Q_diversity": round(q_div, 4),
        "Q_final": round(q_final, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=["mini", "full"])
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    stage = parse_stage(args.stage)
    tasks = {t["task_id"]: t for t in __import__("common").load_jsonl(artifact_path(cfg, "recon_tasks_jsonl")) if t["stage"] == stage}
    src_map = {r["source_instance_row_id"]: r for r in load_csv(artifact_path(cfg, "source_instances_csv"))}
    manifest_path = Path(cfg["project_root"]) / "recon_outputs_tmp" / stage / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    all_rows: List[Dict[str, Any]] = []
    pass_rows: List[Dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            task = tasks[row["task_id"]]
            src = src_map[task["source_instance_row_id"]]
            cid = int(task["class_id"])
            qc = qc_placeholder(cid, float(task["S_pos"]), float(src["Q_src"]), int(row["version"]))
            q_th = cfg["qc_thresholds"]["q_final"].get(cid) or cfg["qc_thresholds"]["q_final"].get(str(cid))
            merged = {
                **row,
                "class_name": task["class_name"],
                "S_pos": task["S_pos"],
                "Q_src": src["Q_src"],
                **qc,
                "Q_threshold": q_th,
                "pass": int(qc["Q_final"] >= float(q_th)),
            }
            all_rows.append(merged)
            if merged["pass"] == 1:
                pass_rows.append(merged)

    save_csv(all_rows, artifact_path(cfg, "recon_qc_all_csv"))
    save_csv(pass_rows, artifact_path(cfg, "recon_qc_pass_csv"))
    print(f"QC done: all={len(all_rows)}, pass={len(pass_rows)}")


if __name__ == "__main__":
    main()
