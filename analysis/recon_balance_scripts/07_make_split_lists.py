from __future__ import annotations

import argparse
from pathlib import Path

from common import image_and_label_dirs, list_images, load_yaml, write_txt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=["mini", "full"])
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    split_root = Path(cfg["split_root"])
    split_root.mkdir(parents=True, exist_ok=True)

    raw_train_images, _ = image_and_label_dirs(cfg, split="train")
    raw_val_images, _ = image_and_label_dirs(cfg, split="val")
    recon_train_images = Path(cfg["recon_pass_root"]) / "images/train"

    raw_train = [str(p) for p in list_images(raw_train_images)]
    raw_val = [str(p) for p in list_images(raw_val_images)]
    recon_train = [str(p) for p in list_images(recon_train_images)] if recon_train_images.exists() else []
    union_train = raw_train + recon_train

    write_txt(raw_train, split_root / "train_raw.txt")
    write_txt(raw_val, split_root / "val_raw.txt")
    write_txt(recon_train, split_root / f"train_recon_{args.stage}.txt")
    write_txt(union_train, split_root / f"train_union_{args.stage}.txt")

    print(f"raw_train={len(raw_train)}, recon_train={len(recon_train)}, union={len(union_train)}")


if __name__ == "__main__":
    main()
