from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n-seg.yaml")  # build a new model from YAML
model = YOLO("weights/yolov8n-seg.pt")  # load a pretrained model (recommended for training)
model.load("weights/yolov8n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="dataset/VGA4/vga.yaml", epochs=30, imgsz=640)