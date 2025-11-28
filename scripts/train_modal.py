"""
Train YOLO on Modal for box detection on conveyor belt.
Auto-labeling + Training pipeline.
"""
import modal

app = modal.App("yolo-box-detector")

# Docker image with YOLO dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "git")  # OpenCV deps + git
    .pip_install(
        "ultralytics>=8.3.0",
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24.0",
        "supervision>=0.25.0",  # For auto-labeling
        "git+https://github.com/ultralytics/CLIP.git",  # For YOLO-World
    )
)

volume = modal.Volume.from_name("yolo-box-detector", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    volumes={"/data": volume},
    timeout=30 * 60,  # 30 min for labeling
)
def auto_label_images():
    """
    Auto-label images using YOLO-World zero-shot detection.
    Detects 'box', 'package', 'carton' on conveyor belt images.
    """
    from ultralytics import YOLO
    import os
    import cv2
    import numpy as np

    print("🏷️  Starting auto-labeling with YOLO-World...")

    images_dir = "/data/dataset/images"
    labels_dir = "/data/dataset/labels"
    os.makedirs(labels_dir, exist_ok=True)

    # Load YOLO-World for zero-shot detection
    model = YOLO("yolov8s-worldv2.pt")

    # Set classes to detect
    model.set_classes(["box", "package", "carton", "cardboard box", "product box"])

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
    print(f"Found {len(image_files)} images to label")

    labeled_count = 0
    total_boxes = 0

    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # Run detection
        results = model(img, conf=0.25, verbose=False)

        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # Convert to YOLO format (normalized cx, cy, w, h)
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                boxes.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if boxes:
            label_path = os.path.join(labels_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
            with open(label_path, 'w') as f:
                f.write('\n'.join(boxes))
            labeled_count += 1
            total_boxes += len(boxes)

        if labeled_count % 10 == 0:
            print(f"  Labeled {labeled_count}/{len(image_files)} images...")

    print(f"\n✅ Auto-labeling complete!")
    print(f"   Images labeled: {labeled_count}/{len(image_files)}")
    print(f"   Total boxes detected: {total_boxes}")

    volume.commit()
    return labeled_count, total_boxes


@app.function(
    image=image,
    gpu="H100",
    volumes={"/data": volume},
    timeout=60 * 60,  # 1 hour
)
def train_yolo(epochs: int = 50, imgsz: int = 640, batch: int = 16):
    """Train YOLO model on conveyor box dataset."""
    from ultralytics import YOLO
    import os

    print("Starting YOLO training on Modal...")

    # Check dataset
    dataset_yaml = "/data/dataset/dataset.yaml"
    if not os.path.exists(dataset_yaml):
        raise FileNotFoundError(f"Dataset not found at {dataset_yaml}. Upload dataset first!")

    # Load base model
    model = YOLO("yolo11n.pt")

    # Train
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="box_detector",
        project="/data/runs/detect",
        patience=15,
        save=True,
        plots=True,
    )

    print("\n✅ Training complete!")
    print(f"Best model saved to: /data/runs/detect/box_detector/weights/best.pt")

    # Commit volume changes
    volume.commit()

    return "/data/runs/detect/box_detector/weights/best.pt"


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=10 * 60,
)
def upload_dataset(dataset_tar_bytes: bytes):
    """Upload and extract dataset to Modal volume."""
    import tarfile
    import io
    import os

    print("Extracting dataset...")

    # Extract tar to /data/dataset
    os.makedirs("/data/dataset", exist_ok=True)

    with tarfile.open(fileobj=io.BytesIO(dataset_tar_bytes), mode="r:gz") as tar:
        tar.extractall("/data/dataset")

    # List contents
    for root, dirs, files in os.walk("/data/dataset"):
        level = root.replace("/data/dataset", "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")

    volume.commit()
    print("\n✅ Dataset uploaded successfully!")


@app.function(
    image=image,
    volumes={"/data": volume},
)
def download_model() -> bytes:
    """Download trained model from Modal volume."""
    import os
    import glob

    # Find the latest training run
    runs_dir = "/data/runs/detect"
    run_dirs = sorted(glob.glob(f"{runs_dir}/box_detector*"))

    if not run_dirs:
        raise FileNotFoundError("No trained models found!")

    # Get the latest run
    latest_run = run_dirs[-1]
    model_path = f"{latest_run}/weights/best.pt"

    print(f"Downloading model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train first!")

    with open(model_path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(action: str = "train", epochs: int = 50):
    """
    Main entrypoint for YOLO training on Modal.

    Usage:
        modal run modal_train.py --action upload    # Upload images first
        modal run modal_train.py --action label     # Auto-label with YOLO-World
        modal run modal_train.py --action train     # Train model
        modal run modal_train.py --action download  # Download trained model
        modal run modal_train.py --action full      # Full pipeline: upload + label + train
    """
    import os
    import tarfile
    import io

    from pathlib import Path
    ROOT = Path(__file__).parent.parent

    if action == "upload":
        print("📦 Preparing dataset for upload...")

        dataset_dir = str(ROOT / "data")

        # Create tar.gz of dataset
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
            # Add images
            images_dir = os.path.join(dataset_dir, "images")
            if os.path.exists(images_dir):
                tar.add(images_dir, arcname="images")

            # Add labels (if any exist)
            labels_dir = os.path.join(dataset_dir, "labels")
            if os.path.exists(labels_dir):
                tar.add(labels_dir, arcname="labels")

            # Add dataset.yaml
            yaml_path = os.path.join(os.path.dirname(dataset_dir), "dataset.yaml")
            if os.path.exists(yaml_path):
                tar.add(yaml_path, arcname="dataset.yaml")

        tar_bytes = tar_buffer.getvalue()
        print(f"📤 Uploading {len(tar_bytes) / 1024 / 1024:.2f} MB...")

        upload_dataset.remote(tar_bytes)
        print("✅ Dataset uploaded!")

    elif action == "label":
        print("🏷️  Starting auto-labeling on Modal...")
        labeled, boxes = auto_label_images.remote()
        print(f"✅ Labeled {labeled} images with {boxes} boxes!")

    elif action == "train":
        print(f"🚀 Starting training with {epochs} epochs on H100...")
        model_path = train_yolo.remote(epochs=epochs)
        print(f"✅ Training complete! Model at: {model_path}")

    elif action == "download":
        print("📥 Downloading trained model...")
        model_bytes = download_model.remote()

        output_path = ROOT / "models" / "best_box_detector.pt"
        with open(output_path, "wb") as f:
            f.write(model_bytes)

        print(f"✅ Model saved to: {output_path}")

    elif action == "full":
        print("🚀 Running full pipeline: upload -> label -> train")

        # Step 1: Upload
        print("\n📦 Step 1: Uploading images...")
        dataset_dir = str(ROOT / "data")
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
            images_dir = os.path.join(dataset_dir, "images")
            if os.path.exists(images_dir):
                tar.add(images_dir, arcname="images")
            yaml_path = str(ROOT / "dataset.yaml")
            if os.path.exists(yaml_path):
                tar.add(yaml_path, arcname="dataset.yaml")
        upload_dataset.remote(tar_buffer.getvalue())
        print("✅ Upload complete!")

        # Step 2: Auto-label
        print("\n🏷️  Step 2: Auto-labeling...")
        labeled, boxes = auto_label_images.remote()
        print(f"✅ Labeled {labeled} images with {boxes} boxes!")

        # Step 3: Train
        print(f"\n🚀 Step 3: Training with {epochs} epochs...")
        model_path = train_yolo.remote(epochs=epochs)
        print(f"✅ Training complete!")

        # Step 4: Download
        print("\n📥 Step 4: Downloading model...")
        model_bytes = download_model.remote()
        output_path = ROOT / "models" / "best_box_detector.pt"
        with open(output_path, "wb") as f:
            f.write(model_bytes)
        print(f"✅ Model saved to: {output_path}")

        print("\n🎉 Full pipeline complete!")

    else:
        print(f"Unknown action: {action}")
        print("Use: upload, label, train, download, or full")
