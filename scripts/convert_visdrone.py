#!/usr/bin/env python3
"""
Convert VisDrone dataset format to YOLO format.

VisDrone format: x, y, w, h, score, class, truncation, occlusion
YOLO format: class, x_center, y_center, width, height (normalized)

Usage:
    python scripts/convert_visdrone.py --data-dir datasets/VisDrone
"""

import argparse
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def visdrone2yolo(data_dir: Path, split: str, source_name: str):
    """Convert VisDrone annotations to YOLO format."""
    source_dir = data_dir / source_name
    images_dir = data_dir / "images" / split
    labels_dir = data_dir / "labels" / split

    # Create output directories
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Move images to new structure
    source_images_dir = source_dir / "images"
    if source_images_dir.exists():
        print(f"Moving images for {split}...")
        for img_path in tqdm(list(source_images_dir.glob("*.jpg")), desc=f"Moving {split} images"):
            shutil.move(str(img_path), str(images_dir / img_path.name))

    # Convert annotations
    annotations_dir = source_dir / "annotations"
    if not annotations_dir.exists():
        print(f"Warning: {annotations_dir} not found, skipping...")
        return

    print(f"Converting annotations for {split}...")
    for ann_file in tqdm(list(annotations_dir.glob("*.txt")), desc=f"Converting {split} labels"):
        img_name = ann_file.with_suffix(".jpg").name
        img_path = images_dir / img_name

        if not img_path.exists():
            print(f"Warning: Image {img_path} not found, skipping {ann_file.name}")
            continue

        # Get image size
        with Image.open(img_path) as img:
            img_width, img_height = img.size

        dw, dh = 1.0 / img_width, 1.0 / img_height
        lines = []

        with open(ann_file, encoding="utf-8") as f:
            for line in f.read().strip().splitlines():
                row = line.split(",")
                if len(row) < 6:
                    continue

                # VisDrone: score=0 means ignored region
                if row[4] == "0":
                    continue

                x, y, w, h = map(int, row[:4])
                cls = int(row[5]) - 1  # VisDrone class starts from 1

                # Convert to YOLO format (normalized)
                x_center = (x + w / 2) * dw
                y_center = (y + h / 2) * dh
                w_norm = w * dw
                h_norm = h * dh

                # Clamp to [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                w_norm = max(0, min(1, w_norm))
                h_norm = max(0, min(1, h_norm))

                lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        # Write YOLO format label
        label_file = labels_dir / ann_file.name
        label_file.write_text("".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Convert VisDrone to YOLO format")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/VisDrone"),
                        help="Path to VisDrone dataset directory")
    parser.add_argument("--clean", action="store_true",
                        help="Remove original VisDrone directories after conversion")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    print(f"Processing dataset at: {data_dir}")

    # Define splits mapping
    splits = {
        "VisDrone2019-DET-train": "train",
        "VisDrone2019-DET-val": "val",
        "VisDrone2019-DET-test-dev": "test",
    }

    # Convert each split
    for source_name, split in splits.items():
        source_dir = data_dir / source_name
        if not source_dir.exists():
            print(f"Skipping {source_name} (not found)")
            continue

        print(f"\n{'='*50}")
        print(f"Processing {source_name} -> {split}")
        print(f"{'='*50}")
        visdrone2yolo(data_dir, split, source_name)

        # Clean up original directory if requested
        if args.clean:
            print(f"Removing {source_dir}...")
            shutil.rmtree(source_dir)

    print(f"\n{'='*50}")
    print("Conversion completed!")
    print(f"{'='*50}")
    print(f"Output structure:")
    print(f"  {data_dir}/images/train/  - {len(list((data_dir/'images'/'train').glob('*.jpg')))} images")
    print(f"  {data_dir}/labels/train/  - {len(list((data_dir/'labels'/'train').glob('*.txt')))} labels")
    print(f"  {data_dir}/images/val/    - {len(list((data_dir/'images'/'val').glob('*.jpg')))} images")
    print(f"  {data_dir}/labels/val/    - {len(list((data_dir/'labels'/'val').glob('*.txt')))} labels")


if __name__ == "__main__":
    main()
