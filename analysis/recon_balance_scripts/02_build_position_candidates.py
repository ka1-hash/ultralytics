from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from common import (
    artifact_path,
    class_name,
    clamp,
    image_and_label_dirs,
    iou,
    load_csv,
    load_image_size,
    load_yaml,
    percentile_rank_normalize,
    read_yolo_labels,
    save_csv,
    Box,
)


def build_candidate_boxes(src_box: Box, img_w: int, img_h: int, class_id: int) -> List[Box]:
    # 左右相邻扫描：继承旧 SI-OCP 思路的简化版本
    gap_min_ratio = {6: 0.15, 7: 0.20, 8: 0.10}[class_id]
    gap_max_ratio = {6: 0.70, 7: 0.60, 8: 1.00}[class_id]
    same_y_ratio = {6: 0.15, 7: 0.10, 8: 0.20}[class_id]
    gap_min = max(4.0, gap_min_ratio * src_box.w)
    gap_max = max(8.0, gap_max_ratio * src_box.w)
    scan_step_x = max(2.0, 0.10 * src_box.w)
    scan_step_y = max(1.0, 0.10 * src_box.h)
    dy_limit = max(2.0, same_y_ratio * src_box.h)

    out: List[Box] = []
    for sign in [-1, 1]:
        gap = gap_min
        while gap <= gap_max + 1e-6:
            base_cx = src_box.cx + sign * (src_box.w / 2.0 + gap + src_box.w / 2.0)
            dy = -dy_limit
            while dy <= dy_limit + 1e-6:
                cy = src_box.cy + dy
                box = Box(base_cx - src_box.w / 2.0, cy - src_box.h / 2.0, base_cx + src_box.w / 2.0, cy + src_box.h / 2.0)
                if box.x1 >= 0 and box.y1 >= 0 and box.x2 <= img_w and box.y2 <= img_h:
                    out.append(box)
                dy += scan_step_y
            gap += scan_step_x
    return out


def score_candidate(cfg: Dict[str, Any], class_id: int, cand: Box, src: Dict[str, Any], gt_boxes: List[Box], img_w: int, img_h: int) -> Dict[str, float]:
    # S_empty
    max_iou = max((iou(cand, g) for g in gt_boxes), default=0.0)
    s_empty = 1.0 - clamp(max_iou / 0.10, 0.0, 1.0)

    # S_same_band
    same_y_ratio = cfg["position_rules"]["same_y_ratio"].get(class_id) or cfg["position_rules"]["same_y_ratio"].get(str(class_id))
    dy_limit = max(2.0, float(same_y_ratio) * float(src["box_h"]))
    s_same = 1.0 - clamp(abs(cand.cy - ((float(src["y1"]) + float(src["y2"])) / 2.0)) / max(dy_limit, 1e-6), 0.0, 1.0)

    # S_scale：目前位置生成直接继承源框大小，给高分；未来可按尺寸模板做更细分
    s_scale = 1.0

    # S_texture / S_density / S_context 先给工程占位值，可替换为真实道路语义 / GDINO 结果
    near_count = sum(1 for g in gt_boxes if abs(g.cy - cand.cy) <= max(cand.h, 16) and abs(g.cx - cand.cx) <= max(cand.w * 1.5, 24))
    s_density = 1.0 if 1 <= near_count <= 4 else (0.7 if near_count == 0 else 0.5)
    s_context = 0.80 if class_id == 8 else (0.78 if class_id == 6 else 0.74)
    edge_margin = min(cand.x1, cand.y1, img_w - cand.x2, img_h - cand.y2)
    s_texture = clamp(edge_margin / max(0.03 * min(img_w, img_h), 1e-6), 0.0, 1.0)

    s_pos = 0.25 * s_empty + 0.20 * s_same + 0.20 * s_context + 0.15 * s_scale + 0.10 * s_texture + 0.10 * s_density
    return {
        "S_empty": round(s_empty, 4),
        "S_same_band": round(s_same, 4),
        "S_context": round(s_context, 4),
        "S_scale": round(s_scale, 4),
        "S_texture": round(s_texture, 4),
        "S_density": round(s_density, 4),
        "S_pos": round(s_pos, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    src_rows = [r for r in load_csv(artifact_path(cfg, "source_instances_csv")) if int(r.get("keep_instance", 0)) == 1]
    images_dir, labels_dir = image_and_label_dirs(cfg, split="train")

    rows: List[Dict[str, Any]] = []
    grouped = defaultdict(list)
    for r in src_rows:
        grouped[r["image_id"]].append(r)

    for image_id, src_list in grouped.items():
        # image lookup
        img_path = None
        for ext in [".jpg", ".png", ".jpeg", ".bmp", ".webp"]:
            p = images_dir / f"{image_id}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue
        img_w, img_h = load_image_size(img_path)
        label_path = labels_dir / f"{image_id}.txt"
        gt_boxes = [r["box"] for r in read_yolo_labels(label_path, img_w, img_h)]

        for src in src_list:
            class_id = int(src["class_id"])
            src_box = Box(float(src["x1"]), float(src["y1"]), float(src["x2"]), float(src["y2"]))
            cands = build_candidate_boxes(src_box, img_w, img_h, class_id)
            s_pos_th = cfg["position_rules"]["s_pos_threshold"].get(class_id) or cfg["position_rules"]["s_pos_threshold"].get(str(class_id))
            top_k = cfg["position_rules"]["top_k_positions"].get(class_id) or cfg["position_rules"]["top_k_positions"].get(str(class_id))
            scored: List[Dict[str, Any]] = []
            for c in cands:
                if any(iou(c, g) > 0 for g in gt_boxes):
                    continue
                score = score_candidate(cfg, class_id, c, src, gt_boxes, img_w, img_h)
                if score["S_pos"] < float(s_pos_th):
                    continue
                scored.append(
                    {
                        "image_id": image_id,
                        "image_path": str(img_path),
                        "source_image_id": src["image_id"],
                        "source_class_id": class_id,
                        "source_class_name": class_name(cfg, class_id),
                        "source_instance_row_id": f"{src['image_id']}::{class_id}::{src['x1']}::{src['y1']}::{src['x2']}::{src['y2']}",
                        "source_x1": src["x1"],
                        "source_y1": src["y1"],
                        "source_x2": src["x2"],
                        "source_y2": src["y2"],
                        "target_x1": round(c.x1, 3),
                        "target_y1": round(c.y1, 3),
                        "target_x2": round(c.x2, 3),
                        "target_y2": round(c.y2, 3),
                        **score,
                    }
                )
            scored.sort(key=lambda x: x["S_pos"], reverse=True)
            rows.extend(scored[: int(top_k)])

    save_csv(rows, artifact_path(cfg, "candidate_positions_csv"))
    print(f"Saved {len(rows)} candidate positions -> {artifact_path(cfg, 'candidate_positions_csv')}")


if __name__ == "__main__":
    main()
