import os
import cv2
import json
import pickle
import numpy as np
from tqdm import tqdm
from ultralytics.models.sam import Predictor as SAMPredictor
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt

def get_annos(s='train', dataset='ssdd'):
    annos = json.load(open(f'/home/datasets/{dataset}/{s}.json', 'r'))
    images = annos['images']
    imgids = [img['id'] for img in images]
    data = {img['id']: {'file_name': img['file_name'], 'annos': []} for img in images}
    for ann in annos['annotations']:
        bbox = ann['bbox']
        data[ann['image_id']]['annos'].append(bbox)
    return data, imgids

def get_predictor(conf=0.25, model='mobile_sam'):
    # overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=512, model="/home/liuwei/projects/research/ultralytics/sam_l.pt", device="cpu")
    overrides = dict(conf=conf, task="segment", mode="predict", save=False, verbose=False, model=f"/home/liuwei/projects/research/ultralytics/{model}.pt")
    predictor = SAMPredictor(overrides=overrides)
    return predictor

def sam_box(data, imgids, predictor, mask_dir):
    for imgid in tqdm(imgids):
        bboxes = np.array(data[int(imgid)]['annos']).astype(int)
        bboxes[:, 2:] += bboxes[:, :2]  # xywh to xyxy
        # img = cv2.imread(os.path.join('/home/datasets/ssdd/JPEGImages', data[int(imgid)]['file_name']))
        img = cv2.imread(os.path.join('/home/datasets/sardet100k/train', data[int(imgid)]['file_name']))
        predictor.set_image(img)  # set with image file
        mask = np.zeros_like(img[..., 0], dtype=np.bool_)
        for bbox in bboxes:
            results = predictor(bboxes=bbox)
            mask |= results[0].masks.cpu().numpy().data[0]
        mask = np.where(mask, 255, 0).astype(np.uint8)
        cv2.imwrite(os.path.join(mask_dir, data[int(imgid)]['file_name'].replace('.jpg', '.png')), mask)
        predictor.reset_image()  # reset image

def sam_box_single(data, imgid, predictor):
    for imgid in tqdm(imgids):
        bboxes = np.array(data[imgid]['annos']).astype(int)
        if len(bboxes) != 3:
            continue
        print(imgid)
        bboxes[:, 2:] += bboxes[:, :2]  # xywh to xyxy
        img = cv2.imread(os.path.join('/home/datasets/ssdd/JPEGImages', data[imgid]['file_name']))
        # predictor.set_image(img)  # set with image file
        # mask = np.zeros_like(img[..., 0], dtype=np.bool_)
        # for i, bbox in enumerate(bboxes):
        #     results = predictor(bboxes=bbox)
        #     mask = results[0].masks.cpu().numpy().data[0]
        #     mask = np.where(mask, 255, 0).astype(np.uint8)
        #     cv2.imwrite(f'{imgid}_{i}.jpg', mask)
        
        # 画一张边框为黑色，背景为白色的图片，box为红色框
        img_with_box = np.ones_like(img, dtype=np.uint8) * 255
        cv2.rectangle(img_with_box, (0, 0), (img.shape[1]-1, img.shape[0]-1), (0, 0, 0), 3)
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (0, 0, 255), 2) # 红色框
        cv2.imwrite(f'{imgid}.jpg', img_with_box)
        break

def sam_center_point(data, imgids, predictor):
    for imgid in tqdm(imgids):
        if imgid != 1033:
            continue
        bboxes = np.array(data[int(imgid)]['annos']).astype(int)
        bboxes[:, 2:] += bboxes[:, :2]
        points = (bboxes[:, :2] + bboxes[:, 2:])/2
        img = cv2.imread(os.path.join('/home/datasets/ssdd/JPEGImages', data[int(imgid)]['file_name']))
        predictor.set_image(img)  # set with image file
        total_mask = np.zeros_like(img[..., 0], dtype=np.bool_)
        pesudo_bboxes = []
        for bbox in bboxes:
            results = predictor(bboxes=bbox)
            mask = results[0].masks.cpu().numpy().data[0]
            total_mask |= mask
            # 最大外接包围框
            mask = np.where(mask, 255, 0).astype(np.uint8)
            x, y, w, h = cv2.boundingRect(mask)
            pesudo_bboxes.append([x, y, x+w, y+h])
        total_mask = np.where(total_mask, 255, 0).astype(np.uint8)
        # 绘制原始框和伪框
        total_mask = total_mask[..., None].repeat(3, -1)
        for bbox, pesudo_bbox in zip(bboxes, pesudo_bboxes):
            x1, y1, x2, y2 = bbox
            x1_p, y1_p, x2_p, y2_p = pesudo_bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(total_mask, (x1_p, y1_p), (x2_p, y2_p), (0, 0, 255), 2)
            combined_img = np.ones((img.shape[0], img.shape[1]*2+10, 3), dtype=np.uint8) * 255
            combined_img[:, :img.shape[1], :] = img
            combined_img[:, img.shape[1]+10:, :] = total_mask
            cv2.imwrite(f'{imgid}.jpg', combined_img)

def sam_grid_point(data, imgids, predictor):
    for imgid in tqdm(imgids):
        if imgid != 2:
            continue
        img = cv2.imread(os.path.join('/home/datasets/ssdd/JPEGImages', data[int(imgid)]['file_name']))
        predictor.set_image(img)  # set with image file
        total_mask = np.zeros_like(img[..., 0], dtype=np.bool_)
        h, w = img.shape[:2]
        points = []
        grid_size = 64
        for y in range(grid_size//2, h, grid_size):
            for x in range(grid_size//2, w, grid_size):
                points.append([x, y])
        results = predictor(points=np.array(points)) # names共40个点，mask40*337*499，每个点对应一个mask
        mask = results[0].masks.cpu().numpy().data
        with open(f'{imgid}.pkl', 'wb') as f:
            pickle.dump(mask, f)


if __name__ == '__main__':
    data, imgids = get_annos(s='train', dataset='sardet100k')
    predictor = get_predictor(model='mobile_sam')

    # sam_grid_point(data, imgids, predictor)
    
    mask_dir = '/home/datasets/sardet100k/mask_sam_mobile'
    os.makedirs(mask_dir, exist_ok=True)
    sam_box(data, imgids, predictor, mask_dir)












# fastsam
# model = FastSAM("/home/liuwei/projects/research/ultralytics/FastSAM-x.pt")
# for imgid in tqdm(imgids):
#     bboxes = np.array(data[int(imgid)]['annos']).astype(int)
#     bboxes[:, 2:] += bboxes[:, :2]  # xywh to xyxy
#     img = cv2.imread(os.path.join('/home/datasets/ssdd/JPEGImages', data[int(imgid)]['file_name']))
#     everything_results = model(img, device="0", retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
#     prompt_process = FastSAMPrompt(img, everything_results, device="0")
#     mask = np.zeros_like(img[..., 0], dtype=np.bool_)
#     for bbox in bboxes:
#         results = prompt_process.box_prompt(bbox=bbox)
#         if results[0].masks is not None:
#             mask |= results[0].masks.cpu().numpy().data[0].astype(np.bool_)
#     mask = np.where(mask, 255, 0).astype(np.uint8)
#     cv2.imwrite(os.path.join(mask_dir, data[int(imgid)]['file_name'].replace('.jpg', '.png')), mask)


