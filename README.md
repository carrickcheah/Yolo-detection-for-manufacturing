# Box Counter

[![Demo](https://img.youtube.com/vi/Mdgg-ZLQbUw/maxresdefault.jpg)](https://www.youtube.com/shorts/Mdgg-ZLQbUw)

Computer vision system for counting boxes on a conveyor belt using line-crossing detection.

## Tech Stack

- **Python 3.12**
- **YOLOv11** (Ultralytics) - Object detection & tracking
- **OpenCV** - Video processing
- **Modal** - Cloud GPU training

## Project Structure

```
box_counter/
├── src/
│   ├── counter.py          # Main counting logic (YOLO + ByteTrack)
│   └── counter_legacy.py   # Background subtraction approach
├── scripts/
│   ├── train_modal.py      # Train on Modal (cloud GPU)
│   ├── train_local.py      # Train locally
│   ├── label.py            # Manual labeling tool
│   └── extract_frames.py   # Extract frames from video
├── data/
│   ├── images/             # Training images
│   ├── labels/             # YOLO format labels
│   └── videos/             # Input/output videos
├── models/                 # Model weights (.pt files)
├── dataset.yaml            # YOLO dataset config
└── pyproject.toml
```

## Quick Start

```bash
# Install
uv sync

# Run counter
python src/counter.py
```

## Training

```bash
# Label data
python scripts/label.py

# Train on Modal (cloud GPU)
modal run scripts/train_modal.py --action full
```
