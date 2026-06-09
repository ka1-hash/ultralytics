from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from common import artifact_path, image_and_label_dirs, load_csv, load_image_size, load_yaml, parse_stage, save_csv, stage_rule


def choose_budget(cfg: Dict, stage: str, raw_label_dir: Path, class_ids: List[int]) -> Dict[int, int]:
    raw_counts = Counter()
    for label_path in raw_label_dir.glob("*.txt"):
        txt = label_path.read_text(encoding="utf-8").strip().splitlines()
        for line in txt:
            if not line.strip() or line.startswith("#"):
                continue
            cls = int(float(line.split()[0]))
            if cls in class_ids:
                raw_counts[cls] += 1
    budgets: Dict[int, int] = {}
    for cid in class_ids:
        lo, hi = stage_rule(cfg, stage, "add_ratio", cid)
        ratio = (float(lo) + float(hi)) / 2.0
        budgets[cid] = round(raw_counts[cid] * ratio)
    return budgets


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> str:
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return f"{xc/img_w:.6f} {yc/img_h:.6f} {bw/img_w:.6f} {bh/img_h:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=["mini", "full"])
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    stage = parse_stage(args.stage)
    pass_rows = load_csv(artifact_path(cfg, "recon_qc_pass_csv"))
    if not pass_rows:
        print("No pass rows found.")
        return

    _, raw_labels_dir = image_and_label_dirs(cfg, split="train")
    class_ids = sorted(int(k) for k in cfg["class_names"].keys()) if isinstance(next(iter(cfg["class_names"].keys())), int) else sorted(int(k) for k in cfg["class_names"].keys())
    budgets = choose_budget(cfg, stage, raw_labels_dir, class_ids)

    by_class = defaultdict(list)
    for r in pass_rows:
        cid = int(r["class_id"])
        by_class[cid].append(r)

    recon_root = Path(cfg["recon_pass_root"])
    out_img = recon_root / "images/train"
    out_lbl = recon_root / "labels/train"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    kept_rows = []
    counts = Counter()
    for cid in class_ids:
        rows = sorted(by_class[cid], key=lambda x: float(x["Q_final"]), reverse=True)
        take = rows[: budgets[cid]]
        for r in take:
            src_img = Path(r["output_image"])
            if not src_img.exists():
                continue
            dst_img = out_img / src_img.name
            shutil.copy2(src_img, dst_img)

            # 当前模板版：标签文件只写新增目标；真实项目中可在此合并原图 GT + 新目标
            target_bbox = r["target_bbox"]
            if isinstance(target_bbox, str):
                try:
                    target_bbox = json.loads(target_bbox)
                except json.JSONDecodeError:
                    target_bbox = eval(target_bbox)
            x1, y1, x2, y2 = map(float, target_bbox)
            img_w, img_h = load_image_size(dst_img)
            yolo_box = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
            (out_lbl / (src_img.stem + ".txt")).write_text(f"{cid} {yolo_box}\n", encoding="utf-8")

            rr = dict(r)
            rr["assembled_image"] = str(dst_img)
            rr["assembled_label"] = str(out_lbl / (src_img.stem + '.txt'))
            kept_rows.append(rr)
            counts[cid] += 1

    save_csv(kept_rows, artifact_path(cfg, "final_class_counts_csv"))
    print("Assembled ReCon pass dataset")
    for cid in class_ids:
        print(f"class {cid}: keep {counts[cid]} / budget {budgets[cid]}")


if __name__ == "__main__":
    main()
