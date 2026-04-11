from ultralytics import YOLO, RTDETR
# Load a model
model = YOLO('yolov8m.pt')
# Train the model
results = model.train(data='ssdd.yaml', epochs=12, imgsz=640, batch=8) # 是一个示例数据集