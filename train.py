from ultralytics import YOLO 


model = YOLO()
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640
)