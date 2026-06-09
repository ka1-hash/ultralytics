from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any, Dict, List

from common import (
    artifact_path,
    class_name,
    clamp,
    image_and_label_dirs,
    load_image_size,
    load_yaml,
    percentile_rank_normalize,
    read_yolo_labels,
    save_csv,
    score_centered_range,
    short_train,
)


def fake_sam_metrics(class_id: int, box_w: float, box_h: float) -> Dict[str, float]:
    """占位函数：请替换成真实 SAM 调用。

    当前返回一组稳定的几何近似值，方便先跑通统计与流程。
    """
    aspect = max(box_w, box_h) / max(min(box_w, box_h), 1e-6)
    area_ratio = 0.45 if class_id == 6 else (0.50 if class_id == 7 else 0.62)
    area_ratio = clamp(area_ratio - 0.03 * min(aspect - 1.0, 3.0), 0.2, 0.95)
    return {
        "mask_bbox_iou": 0.72 if class_id == 7 else 0.70,
        "mask_area_ratio": area_ratio,
        "largest_cc_ratio": 0.82 if class_id in [7, 8] else 0.76,
        "cc_count": 1,
    }


def pass_s4(cfg: Dict[str, Any], class_id: int, sam_metrics: Dict[str, float]) -> bool:
    rules = cfg["s4_rules"].get(class_id) or cfg["s4_rules"].get(str(class_id))
    if not rules:
        return False
    return (
        sam_metrics["mask_bbox_iou"] >= float(rules["mask_bbox_iou_min"])
        and float(rules["mask_area_ratio_min"]) <= sam_metrics["mask_area_ratio"] <= float(rules["mask_area_ratio_max"])
        and sam_metrics["largest_cc_ratio"] >= float(rules["largest_cc_ratio_min"])
        and sam_metrics["cc_count"] <= int(rules["cc_count_max"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    images_dir, labels_dir = image_and_label_dirs(cfg, split="train")
    target_classes = {int(k) for k in cfg["class_names"].keys()} if isinstance(next(iter(cfg["class_names"].keys())), int) else {int(k) for k in cfg["class_names"].keys()}

    raw_rows: List[Dict[str, Any]] = []

    # 第一遍：收集基础几何、SAM 占位指标、边缘占位值
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        label_path = labels_dir / (image_path.stem + ".txt")
        img_w, img_h = load_image_size(image_path)
        labels = read_yolo_labels(label_path, img_w, img_h)
        boxes = [row["box"] for row in labels]
        for idx, row in enumerate(labels):
            class_id = int(row["class_id"])
            if class_id not in target_classes:
                continue
            box = row["box"]
            st = short_train(box, img_w, img_h, 1280)
            lo, hi = (cfg["s2_short_train_ranges"].get(class_id) or cfg["s2_short_train_ranges"].get(str(class_id)))
            boundary_margin = min(box.x1, box.y1, img_w - box.x2, img_h - box.y2)
            neighbor_count = 0
            max_neighbor_iou = 0.0
            for j, other in enumerate(labels):
                if j == idx:
                    continue
                obox = other["box"]
                # 简化邻域定义：中心点落在外扩 30% 区域内
                expand_x = 0.3 * box.w
                expand_y = 0.3 * box.h
                if (box.x1 - expand_x) <= obox.cx <= (box.x2 + expand_x) and (box.y1 - expand_y) <= obox.cy <= (box.y2 + expand_y):
                    neighbor_count += 1
                # 近似最大重叠
                ix1 = max(box.x1, obox.x1)
                iy1 = max(box.y1, obox.y1)
                ix2 = min(box.x2, obox.x2)
                iy2 = min(box.y2, obox.y2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                union = box.area + obox.area - inter
                max_neighbor_iou = max(max_neighbor_iou, 0.0 if union <= 0 else inter / union)

            # 占位边缘强度：尺寸越合理越高；后续请替换为真实梯度统计
            edge_strength = min(box.w, box.h) / max(max(box.w, box.h), 1e-6)
            edge_strength = 0.6 + 0.4 * edge_strength
            sam_metrics = fake_sam_metrics(class_id, box.w, box.h)

            raw_rows.append(
                {
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "image_id": image_path.stem,
                    "class_id": class_id,
                    "class_name": class_name(cfg, class_id),
                    "x1": round(box.x1, 3),
                    "y1": round(box.y1, 3),
                    "x2": round(box.x2, 3),
                    "y2": round(box.y2, 3),
                    "box_w": round(box.w, 3),
                    "box_h": round(box.h, 3),
                    "short_train": round(st, 3),
                    "s2_low": lo,
                    "s2_high": hi,
                    "s2_pass": int(lo <= st <= hi),
                    "boundary_margin": round(boundary_margin, 3),
                    "neighbor_count": neighbor_count,
                    "max_neighbor_iou": round(max_neighbor_iou, 4),
                    "edge_strength": round(edge_strength, 4),
                    "mask_bbox_iou": sam_metrics["mask_bbox_iou"],
                    "mask_area_ratio": sam_metrics["mask_area_ratio"],
                    "largest_cc_ratio": sam_metrics["largest_cc_ratio"],
                    "cc_count": sam_metrics["cc_count"],
                    "s4_pass": int(pass_s4(cfg, class_id, sam_metrics)),
                }
            )

    save_csv(raw_rows, artifact_path(cfg, "source_instances_raw_csv"))

    # 第二遍：基于分位值归一化并计算 Q_src
    edge_by_class = {}
    boundary_by_class = {}
    for cid in target_classes:
        sub = [r for r in raw_rows if int(r["class_id"]) == cid]
        if not sub:
            continue
        edge_vals = [float(r["edge_strength"]) for r in sub]
        boundary_vals = [float(r["boundary_margin"]) for r in sub]
        edge_by_class[cid] = (statistics.quantiles(edge_vals, n=10)[0], statistics.quantiles(edge_vals, n=10)[-1]) if len(edge_vals) >= 10 else (min(edge_vals), max(edge_vals))
        boundary_by_class[cid] = (statistics.quantiles(boundary_vals, n=10)[0], statistics.quantiles(boundary_vals, n=10)[-1]) if len(boundary_vals) >= 10 else (min(boundary_vals), max(boundary_vals))

    rows: List[Dict[str, Any]] = []
    for r in raw_rows:
        cid = int(r["class_id"])
        lo = float(r["s2_low"])
        hi = float(r["s2_high"])
        q_size = score_centered_range(float(r["short_train"]), lo, hi)

        area_ratio = float(r["mask_area_ratio"])
        rules = cfg["s4_rules"].get(cid) or cfg["s4_rules"].get(str(cid))
        ar_low = float(rules["mask_area_ratio_min"])
        ar_high = float(rules["mask_area_ratio_max"])
        area_score = score_centered_range(area_ratio, ar_low, ar_high)
        cc_score = 0.5 * clamp((float(r["largest_cc_ratio"]) - 0.5) / 0.5, 0.0, 1.0) + 0.5 * (1.0 if int(r["cc_count"]) <= int(rules["cc_count_max"]) else 0.0)
        q_mask = 0.5 * float(r["mask_bbox_iou"]) + 0.3 * area_score + 0.2 * cc_score

        p10, p90 = edge_by_class[cid]
        q_edge = percentile_rank_normalize(float(r["edge_strength"]), p10, p90)

        q_isolation = clamp(1.0 - 0.6 * min(int(r["neighbor_count"]) / 3.0, 1.0) - 0.4 * min(float(r["max_neighbor_iou"]) / 0.3, 1.0), 0.0, 1.0)
        bp10, bp90 = boundary_by_class[cid]
        q_boundary = percentile_rank_normalize(float(r["boundary_margin"]), bp10, bp90)

        # 域一致性占位：默认原始图全部给 0.8，可在后续替换为亮度/路面/来源评分
        q_domain = 0.8

        q_src = 0.25 * q_size + 0.20 * q_mask + 0.20 * q_edge + 0.15 * q_isolation + 0.10 * q_boundary + 0.10 * q_domain
        q_th = cfg["qsrc_thresholds"].get(cid) or cfg["qsrc_thresholds"].get(str(cid))
        keep = int(int(r["s2_pass"]) == 1 and int(r["s4_pass"]) == 1 and q_src >= float(q_th))

        rr = dict(r)
        rr.update(
            {
                "Q_size": round(q_size, 4),
                "Q_mask": round(q_mask, 4),
                "Q_edge": round(q_edge, 4),
                "Q_isolation": round(q_isolation, 4),
                "Q_boundary": round(q_boundary, 4),
                "Q_domain": round(q_domain, 4),
                "Q_src": round(q_src, 4),
                "Q_src_threshold": q_th,
                "keep_instance": keep,
            }
        )
        rows.append(rr)

    save_csv(rows, artifact_path(cfg, "source_instances_csv"))
    kept = sum(int(r["keep_instance"]) for r in rows)
    print(f"Saved {len(rows)} raw instances, kept {kept} instances -> {artifact_path(cfg, 'source_instances_csv')}")


if __name__ == "__main__":
    main()
