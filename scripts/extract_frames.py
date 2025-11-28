"""
Extract frames from video for labeling
"""
import cv2
import os

video_path = "/Users/carrickcheah/Project/root_ai/ultralytics/item.mp4"
output_dir = "/Users/carrickcheah/Project/root_ai/box_counter/dataset/images"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Extract every 10th frame (about 50 frames from 541 total)
frame_interval = 10
count = 0
saved = 0

print(f"Extracting frames from video ({total_frames} total frames)...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if count % frame_interval == 0:
        filename = f"{output_dir}/frame_{count:04d}.jpg"
        cv2.imwrite(filename, frame)
        saved += 1
        print(f"Saved: {filename}")

    count += 1

cap.release()
print(f"\nDone! Extracted {saved} frames to: {output_dir}")
