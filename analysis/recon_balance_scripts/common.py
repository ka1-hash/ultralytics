from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


def load_yaml(path: str | Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML not installed. Please install pyyaml.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_csv(rows: Iterable[Dict[str, Any]], path: str | Path) -> None:
    rows = list(rows)
    ensure_parent(path)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with open(p, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_jsonl(rows: Iterable[Dict[str, Any]], path: str | Path) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def list_images(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])


def yolo_to_xyxy(xc: float, yc: float, bw: float, bh: float, img_w: int, img_h: int) -> Box:
    x1 = (xc - bw / 2.0) * img_w
    y1 = (yc - bh / 2.0) * img_h
    x2 = (xc + bw / 2.0) * img_w
    y2 = (yc + bh / 2.0) * img_h
    return Box(x1, y1, x2, y2)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def iou(box1: Box, box2: Box) -> float:
    ix1 = max(box1.x1, box2.x1)
    iy1 = max(box1.y1, box2.y1)
    ix2 = min(box1.x2, box2.x2)
    iy2 = min(box1.y2, box2.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = box1.area + box2.area - inter
    return 0.0 if union <= 0 else inter / union


def read_yolo_labels(label_path: Path, img_w: int, img_h: int) -> List[Dict[str, Any]]:
    if not label_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    txt = label_path.read_text(encoding="utf-8").strip().splitlines()
    for line in txt:
        if not line.strip():
            continue
        parts = line.split()
        cls = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])
        box = yolo_to_xyxy(xc, yc, bw, bh, img_w, img_h)
        rows.append({
            "class_id": cls,
            "xc": xc,
            "yc": yc,
            "bw": bw,
            "bh": bh,
            "box": box,
        })
    return rows


def short_train(box: Box, img_w: int, img_h: int, train_long_side: int = 1280) -> float:
    long_side = max(img_w, img_h)
    return min(box.w, box.h) * train_long_side / float(long_side)


def score_centered_range(value: float, low: float, high: float) -> float:
    if value < low or value > high:
        return 0.0
    mid = (low + high) / 2.0
    half = (high - low) / 2.0
    if half <= 0:
        return 0.0
    return clamp(1.0 - abs(value - mid) / half, 0.0, 1.0)


def percentile_rank_normalize(value: float, p10: float, p90: float) -> float:
    if p90 <= p10:
        return 1.0
    return clamp((value - p10) / (p90 - p10), 0.0, 1.0)


def parse_stage(stage: str) -> str:
    stage = stage.strip().lower()
    if stage not in {"mini", "full"}:
        raise ValueError("stage must be mini or full")
    return stage


def image_and_label_dirs(cfg: Dict[str, Any], split: str = "train") -> Tuple[Path, Path]:
    root = Path(cfg["raw_dataset_root"])
    if split == "train":
        return root / cfg["images_train_subdir"], root / cfg["labels_train_subdir"]
    return root / cfg["images_val_subdir"], root / cfg["labels_val_subdir"]


def artifact_path(cfg: Dict[str, Any], key: str) -> Path:
    return Path(cfg["project_root"]) / cfg["artifacts"][key]


def class_name(cfg: Dict[str, Any], class_id: int) -> str:
    mapping = cfg["class_names"]
    return mapping.get(class_id) or mapping.get(str(class_id)) or str(class_id)


def stage_rule(cfg: Dict[str, Any], stage: str, section: str, class_id: int) -> Any:
    sec = cfg["c_rules"][f"stage_{stage}"][section]
    return sec.get(class_id) or sec.get(str(class_id))


def load_image_size(image_path: Path) -> Tuple[int, int]:
    from PIL import Image
    with Image.open(image_path) as im:
        return im.size


def write_txt(lines: Iterable[str], path: str | Path) -> None:
    ensure_parent(path)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
