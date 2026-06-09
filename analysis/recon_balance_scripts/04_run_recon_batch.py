from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw

from common import artifact_path, load_jsonl, load_yaml, parse_stage


def render_placeholder(task: Dict[str, Any], out_path: Path) -> None:
    """占位实现：真实项目中请在这里接入 ReCon / ControlNet / SAM / GDINO。

    当前只是把目标框画到原图上，方便整个工程链先跑通。
    """
    src_img = Image.open(task["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(src_img)
    x1, y1, x2, y2 = task["target_bbox"]
    color = {6: (255, 120, 0), 7: (0, 180, 255), 8: (0, 255, 120)}[int(task["class_id"])]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    draw.text((x1 + 2, max(0, y1 - 12)), f"recon_{task['class_id']}", fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    src_img.save(out_path, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=["mini", "full"])
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    stage = parse_stage(args.stage)
    tasks = [t for t in load_jsonl(artifact_path(cfg, "recon_tasks_jsonl")) if t["stage"] == stage]

    out_root = Path(cfg["project_root"]) / "recon_outputs_tmp" / stage
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, Any]] = []
    for task in tasks:
        for version in range(int(task["num_versions"])):
            img_name = f"{task['task_id']}_v{version:02d}.jpg"
            out_path = out_root / img_name
            render_placeholder(task, out_path)
            manifest.append(
                {
                    "task_id": task["task_id"],
                    "version": version,
                    "output_image": str(out_path),
                    "target_image_id": task["target_image_id"],
                    "class_id": task["class_id"],
                    "target_bbox": task["target_bbox"],
                    "source_instance_row_id": task["source_instance_row_id"],
                }
            )

    manifest_path = out_root / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Rendered {len(manifest)} outputs -> {out_root}")


if __name__ == "__main__":
    main()
