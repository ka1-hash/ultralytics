import re
import argparse
from datetime import datetime
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='yolo26n', help='model name, e.g. yolo26n, yolo26m')
parser.add_argument('--data', type=str, default='VisDrone', help='dataset name, e.g. VisDrone, coco')
parser.add_argument('--batch', type=int, default=4, help='total batch size (split across GPUs)')
parser.add_argument('--device', type=str, default='0', help='CUDA device(s), e.g. 0, 0,1,2,3, cpu')
parser.add_argument('--resume', type=str, default='', help='path to last.pt for resuming training')
parser.add_argument('--epochs', type=int, default=120, help='number of epochs')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--imgsz', type=int, default=1280, help='input image size')
parser.add_argument('--multi_scale', type=float, default=0.5, help='multi-scale range fraction')
args = parser.parse_args()

# Resume mode: load checkpoint and continue training
if args.resume:
    model = YOLO(args.resume)
    results = model.train(resume=True)  # resume=True 会自动使用 self.ckpt_path
else:
    MODEL = args.model
    DATA = args.data

    # Load a model
    # P2/P6等变体没有单独的预训练权重，从yaml构建模型结构，再加载基础模型权重
    model_suffix = re.split(r'[-]', MODEL, 1)  # yolo26m-p2 -> ['yolo26m', 'p2']
    base_model = model_suffix[0]  # yolo26m
    variant = model_suffix[1] if len(model_suffix) > 1 else None  # p2 or None
    
    if variant:
        # 变体模型：从yaml构建结构，加载基础模型预训练权重（自动匹配共有层）
        # 传入带scale的文件名如 yolo26m-p2.yaml，内部会自动去掉scale找到yolo26-p2.yaml
        model = YOLO(f'{base_model}-{variant}.yaml')
        model.load(f'weights/{base_model}.pt')
    else:
        # 标准模型：直接加载预训练权重
        model = YOLO(f'weights/{MODEL}.pt')

    # Train the model
    results = model.train(
        data=f'{DATA}.yaml', 
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        multi_scale=args.multi_scale,
        batch=args.batch,
        optimizer='MuSGD',
        lr0=0.01,
        cos_lr=True,
        project=DATA.lower(),
        name=f"{MODEL}-b{args.batch}-s{args.imgsz}-ms{args.multi_scale}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        cache='disk',
        device=args.device,
    )
