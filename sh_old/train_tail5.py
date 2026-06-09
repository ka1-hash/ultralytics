import re
import argparse
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO


def parse_imgsz(x):
    x = str(x).lower().replace(" ", "")
    if "x" in x:
        h, w = x.split("x")
        return [int(h), int(w)]
    if "," in x:
        h, w = x.split(",")
        return [int(h), int(w)]
    return int(x)


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='yolo26n-nop5', help='model name, e.g. yolo26n-nop5')
parser.add_argument('--data', type=str, default='VisDrone_ReCon', help='dataset name')
parser.add_argument('--batch', type=int, default=4, help='total batch size (split across GPUs)')
parser.add_argument('--device', type=str, default='0,1', help='CUDA device(s), e.g. 0, 0,1')
parser.add_argument('--resume', type=str, default='', help='path to last.pt for resuming training')
parser.add_argument('--epochs', type=int, default=120, help='number of epochs')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--imgsz', type=str, default='1280', help='input image size')
parser.add_argument('--multi_scale', type=float, default=0, help='multi-scale range fraction')
parser.add_argument('--weights_init', type=str, default='', help='initialize from an existing best.pt / last.pt')
parser.add_argument('--freeze', type=int, default=0, help='freeze first N layers (ultralytics train arg)')
parser.add_argument('--lr0', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--optimizer', type=str, default='MuSGD', help='optimizer name')
parser.add_argument('--name_prefix', type=str, default='', help='optional run name prefix')
args = parser.parse_args()

IMGSZ = parse_imgsz(args.imgsz)
IMGSZ_NAME = str(args.imgsz).replace(",", "x").replace(" ", "")

if args.resume:
    model = YOLO(args.resume)
    results = model.train(resume=True)
elif args.weights_init:
    model = YOLO(args.weights_init)
    DATA = args.data
    data_arg = DATA if (DATA.endswith('.yaml') or '/' in DATA) else f'{DATA}.yaml'
    data_name = Path(data_arg).stem if (data_arg.endswith('.yaml') or '/' in data_arg) else DATA
    run_name = f"{args.name_prefix}-s{IMGSZ_NAME}-ms{args.multi_scale}-{datetime.now().strftime('%Y%m%d-%H%M%S')}" if args.name_prefix else f"finetune-b{args.batch}-s{IMGSZ_NAME}-ms{args.multi_scale}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    results = model.train(
        data=data_arg,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=IMGSZ,
        multi_scale=args.multi_scale,
        batch=args.batch,
        optimizer=args.optimizer,
        lr0=args.lr0,
        cos_lr=False,
        mosaic=1.0,
        close_mosaic=10,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        project=data_name.lower(),
        name=run_name,
        cache='disk',
        device=args.device,
        freeze=args.freeze if args.freeze > 0 else None,
    )
else:
    MODEL = args.model
    DATA = args.data
    data_arg = DATA if (DATA.endswith('.yaml') or '/' in DATA) else f'{DATA}.yaml'
    data_name = Path(data_arg).stem if (data_arg.endswith('.yaml') or '/' in data_arg) else DATA

    model_suffix = re.split(r'[-]', MODEL, 1)
    base_model = model_suffix[0]
    variant = model_suffix[1] if len(model_suffix) > 1 else None

    if variant:
        model = YOLO(f'{base_model}-{variant}.yaml')
        model.load(f'weights/{base_model}.pt')
    else:
        model = YOLO(f'weights/{MODEL}.pt')

    results = model.train(
        data=data_arg,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=IMGSZ,
        multi_scale=args.multi_scale,
        batch=args.batch,
        optimizer=args.optimizer,
        lr0=args.lr0,
        cos_lr=False,

        mosaic=1.0,
        close_mosaic=10,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,

        project=data_name.lower(),
        name=(f"{args.name_prefix}-" if args.name_prefix else '') + f"{MODEL}-b{args.batch}-s{IMGSZ_NAME}-ms{args.multi_scale}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        cache='disk',
        device=args.device,
        freeze=args.freeze if args.freeze > 0 else None,
    )
