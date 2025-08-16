from ultralytics import YOLO

# Load pretrained model
model = YOLO("best.pt")  # path to your current weights

# Retrain on collected dataset
model.train(
    data="config.yaml",   # dataset config (we'll create this below)
    epochs=20,
    imgsz=640,
    batch=16,
    name="garbage_retrain",
    resume=False
)
