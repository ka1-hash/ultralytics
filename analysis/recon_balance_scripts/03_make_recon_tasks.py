from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from common import artifact_path, class_name, load_csv, load_yaml, parse_stage, save_jsonl, stage_rule


def prompts_for_class(class_id: int) -> Dict[str, str]:
    if class_id == 6:
        return {
            "positive": "a small tricycle in aerial view, realistic UAV image, on urban road, natural shadow",
            "negative": "motorcycle, bicycle, bus, car, blurred, deformed, extra vehicle, artifact",
        }
    if class_id == 7:
        return {
            "positive": "a small covered tricycle with canopy in aerial view, realistic UAV image, on urban road, clear canopy structure",
            "negative": "ordinary tricycle, motorcycle, bicycle, missing canopy, deformed vehicle, extra vehicle, artifact",
        }
    return {
        "positive": "a bus in aerial view, realistic UAV image, on road, rectangular body, natural shadow",
        "negative": "truck, van, car, deformed bus, extra vehicle, artifact",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=["mini", "full"])
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    stage = parse_stage(args.stage)
    cands = load_csv(artifact_path(cfg, "candidate_positions_csv"))

    # 预算按 add_ratio 粗略分配：优先保留高 S_pos 候选，后续真正配额在 assemble 阶段再严格控制
    versions = cfg["recon_rules"]["versions_per_candidate"][stage]
    tasks: List[Dict[str, Any]] = []

    for i, row in enumerate(cands):
        class_id = int(row["source_class_id"])
        patch_size = cfg["recon_rules"]["patch_sizes"].get(class_id) or cfg["recon_rules"]["patch_sizes"].get(str(class_id))
        mode = cfg["recon_rules"]["modes"].get(class_id) or cfg["recon_rules"]["modes"].get(str(class_id))
        prompts = prompts_for_class(class_id)
        tasks.append(
            {
                "task_id": f"{stage}_{i:06d}",
                "stage": stage,
                "source_image_id": row["source_image_id"],
                "target_image_id": row["image_id"],
                "class_id": class_id,
                "class_name": class_name(cfg, class_id),
                "source_instance_row_id": row["source_instance_row_id"],
                "source_bbox": [float(row["source_x1"]), float(row["source_y1"]), float(row["source_x2"]), float(row["source_y2"])],
                "target_bbox": [float(row["target_x1"]), float(row["target_y1"]), float(row["target_x2"]), float(row["target_y2"])],
                "image_path": row["image_path"],
                "patch_size": int(patch_size),
                "recon_mode": mode,
                "positive_prompt": prompts["positive"],
                "negative_prompt": prompts["negative"],
                "num_versions": int(versions),
                "S_pos": float(row["S_pos"]),
            }
        )

    save_jsonl(tasks, artifact_path(cfg, "recon_tasks_jsonl"))
    print(f"Saved {len(tasks)} tasks -> {artifact_path(cfg, 'recon_tasks_jsonl')}")


if __name__ == "__main__":
    main()
