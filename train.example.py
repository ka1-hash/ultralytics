import argparse
from datetime import datetime
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='yolo26n', help='model name, e.g. yolo26n, yolo26m')
parser.add_argument('--data', type=str, default='VisDrone', help='dataset name, e.g. VisDrone, coco')
parser.add_argument('--batch', type=int, default=4, help='total batch size (split across GPUs)')
parser.add_argument('--device', type=str, default='0', help='CUDA device(s), e.g. 0, 0,1,2,3, cpu')
args = parser.parse_args()

MODEL = args.model
DATA = args.data

# Load a model
model = YOLO(f'weights/{MODEL}.pt')

# Train the model
results = model.train(
    data=f'{DATA}.yaml', 
    epochs=120,
    patience=10,
    imgsz=1280,
    multi_scale=0.5,
    batch=args.batch,
    optimizer='MuSGD',
    lr0=0.01,
    cos_lr=False,
    project=DATA.lower(),
    name=f"{MODEL}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    cache='disk',
    device=args.device,
)
