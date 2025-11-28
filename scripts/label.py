"""
Quick labeling tool for box detection
Click to draw bounding boxes, press 's' to save, 'n' for next, 'u' to undo
"""
import cv2
import os
import glob
from pathlib import Path

# Paths
ROOT = Path(__file__).parent.parent
IMAGE_DIR = str(ROOT / "data" / "images")
LABEL_DIR = str(ROOT / "data" / "labels")

os.makedirs(LABEL_DIR, exist_ok=True)

# Get images
images = sorted(glob.glob(f"{IMAGE_DIR}/*.jpg"))
print(f"Found {len(images)} images")

# Find first unlabeled image
current_idx = 0
for i, img_path in enumerate(images):
    label_path = img_path.replace("/images/", "/labels/").replace(".jpg", ".txt")
    if not os.path.exists(label_path):
        current_idx = i
        print(f"Starting at first unlabeled image: {os.path.basename(img_path)}")
        break
boxes = []
drawing = False
start_point = None

def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, boxes, img_display

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_display = img.copy()
        cv2.rectangle(img_display, start_point, (x, y), (0, 255, 0), 2)
        # Draw existing boxes
        for box in boxes:
            cv2.rectangle(img_display, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = start_point
        x2, y2 = x, y
        # Normalize
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if abs(x2-x1) > 10 and abs(y2-y1) > 10:  # Min size
            boxes.append([x1, y1, x2, y2])
            print(f"Box added: {boxes[-1]}")

def save_labels(img_path, boxes, img_shape):
    h, w = img_shape[:2]
    label_path = img_path.replace("/images/", "/labels/").replace(".jpg", ".txt")

    with open(label_path, "w") as f:
        for box in boxes:
            x1, y1, x2, y2 = box
            # Convert to YOLO format (class, cx, cy, w, h) normalized
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"Saved {len(boxes)} boxes to {label_path}")

# Main loop
cv2.namedWindow("Labeler")
cv2.setMouseCallback("Labeler", mouse_callback)

while current_idx < len(images):
    img_path = images[current_idx]
    img = cv2.imread(img_path)
    img_display = img.copy()
    boxes = []

    # Check if labels exist
    label_path = img_path.replace("/images/", "/labels/").replace(".jpg", ".txt")
    if os.path.exists(label_path):
        # Load existing labels
        h, w = img.shape[:2]
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, cx, cy, bw, bh = map(float, parts)
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    boxes.append([x1, y1, x2, y2])

    print(f"\nImage {current_idx+1}/{len(images)}: {os.path.basename(img_path)}")
    print("Controls: s=save, n=next, p=prev, u=undo, q=quit")

    while True:
        # Draw boxes
        display = img_display.copy()
        for i, box in enumerate(boxes):
            cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
            cv2.putText(display, str(i+1), (box[0], box[1]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.putText(display, f"Image {current_idx+1}/{len(images)} | Boxes: {len(boxes)}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Labeler", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            save_labels(img_path, boxes, img.shape)
        elif key == ord('n'):
            current_idx += 1
            break
        elif key == ord('p'):
            current_idx = max(0, current_idx - 1)
            break
        elif key == ord('u'):
            if boxes:
                boxes.pop()
                img_display = img.copy()
                print("Undo last box")
        elif key == ord('q'):
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()
print("\nDone labeling!")
