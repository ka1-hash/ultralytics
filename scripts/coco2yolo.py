import os
import json
import shutil
import cv2
import numpy as np
from collections import defaultdict
from ultralytics.utils import LOGGER, TQDM

# Create dataset directory
orig_dir = '/home/liuwei/projects/datasets/ssdd'
save_dir = '/home/liuwei/projects/datasets/ssdd_yolo'
for p in f'{save_dir}/labels', f'{save_dir}/images':
    os.makedirs(p, exist_ok=True)

for json_file in ['train.json', 'test.json']:
    lname = json_file.split('.')[0]
    img_dir = f'{save_dir}/images/{lname}'
    os.makedirs(img_dir, exist_ok=True)
    fn = f'{save_dir}/labels/{lname}'
    os.makedirs(fn, exist_ok=True)
    with open(f'{orig_dir}/{json_file}') as f:
        data = json.load(f)
    images = {f'{x["id"]:d}': x for x in data["images"]}
    imgToAnns = defaultdict(list)
    for ann in data["annotations"]:
        imgToAnns[ann["image_id"]].append(ann)
    image_txt = []
    # Write labels file
    for img_id, anns in TQDM(imgToAnns.items(), desc=f"Annotations {json_file}"):
        img = images[f"{img_id:d}"]
        h, w = img["height"], img["width"]
        f = img["file_name"]
        # shutil.copy(f'{orig_dir}/JPEGImages/{f}', f'{img_dir}/{f}')
        bboxes = []
        for ann in anns:
            box = np.array(ann["bbox"], dtype=np.float64)
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y
            if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                continue
            # cls = ann["category_id"] - 1
            cls = ann["category_id"]
            box = [cls] + box.tolist()
            if box not in bboxes:
                bboxes.append(box)
        with open(f'{fn}/{f[:-3]}txt', 'a') as file:
            for i in range(len(bboxes)):
                line = ' '.join([str(n) for n in bboxes[i]])
                file.write(line + "\n")

LOGGER.info(f"COCO data converted successfully.\nResults saved to {save_dir}")
