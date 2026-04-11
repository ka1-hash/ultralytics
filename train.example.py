from datetime import datetime
from ultralytics import YOLO

# Load a model
model = YOLO('weights/yolo26n.pt')

# Train the model
results = model.train(
    data='VisDrone.yaml', 
    epochs=120,
    patience=10,
    imgsz=1280,
    multi_scale=0.5,
    batch=4,
    optimizer='MuSGD',
    lr0=0.01,
    cos_lr=False,
    project='visdrone',      # 项目根目录
    name=f"vis-26n-{datetime.now().strftime('%Y%m%d-%H%M%S')}",  # 实验名称（自定义 run-id）
    cache=True, # 图片放到内存中
)
