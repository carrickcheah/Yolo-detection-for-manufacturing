"""
Train YOLO to detect boxes on conveyor
"""
from ultralytics import YOLO

# Load base model
model = YOLO("yolo11n.pt")

# Train
results = model.train(
    data="/Users/carrickcheah/Project/root_ai/box_counter/dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="box_detector",
    patience=10,
)

print("\nTraining complete!")
print("Model saved to: runs/detect/box_detector/weights/best.pt")
