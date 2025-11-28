# Box Counter

[![Demo](https://img.youtube.com/vi/Mdgg-ZLQbUw/maxresdefault.jpg)](https://www.youtube.com/shorts/Mdgg-ZLQbUw)

Real-time object counting system for industrial conveyor belts using YOLOv11 and ByteTrack.

## Features

- **Real-time Detection** - YOLOv11 object detection at 30+ FPS
- **Persistent Tracking** - ByteTrack multi-object tracking with ID persistence
- **Line Crossing Counter** - Counts objects crossing a configurable boundary line
- **Cloud Training** - One-command training pipeline on Modal (H100 GPU)
- **Auto-labeling** - Zero-shot labeling with YOLO-World

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Video     │───▶│   YOLOv11   │───▶│  ByteTrack  │───▶│    Line     │
│   Input     │    │  Detection  │    │   Tracker   │    │   Counter   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                          │                  │                  │
                          ▼                  ▼                  ▼
                    Bounding Boxes     Track IDs          Final Count
```

## Project Structure

```
box_counter/
├── src/
│   ├── counter.py          # Main pipeline (YOLO + ByteTrack + LineCounter)
│   └── counter_legacy.py   # Background subtraction approach
├── scripts/
│   ├── train_modal.py      # Cloud training on Modal (H100)
│   ├── train_local.py      # Local training
│   ├── label.py            # Manual labeling tool
│   └── extract_frames.py   # Video frame extraction
├── data/
│   ├── images/             # Training images
│   ├── labels/             # YOLO format annotations
│   └── videos/             # Input/output videos
├── models/                 # Trained weights (.pt)
├── dataset.yaml            # YOLO dataset config
└── pyproject.toml
```

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- [Modal](https://modal.com) account (for cloud training)

### Installation

```bash
git clone https://github.com/your-username/box_counter.git
cd box_counter
uv sync
```

### Run Counter

```bash
# Place input video
cp your_video.mp4 data/videos/input.mp4

# Run
python src/counter.py
```

Output saved to `data/videos/output.mp4`

## Training Pipeline

### Option 1: Full Pipeline (Recommended)

```bash
# Extract frames from video
python scripts/extract_frames.py

# Run full pipeline: upload → auto-label → train → download
modal run scripts/train_modal.py --action full --epochs 50
```

### Option 2: Step by Step

```bash
# 1. Manual labeling (optional - for better accuracy)
python scripts/label.py

# 2. Upload dataset to Modal
modal run scripts/train_modal.py --action upload

# 3. Auto-label with YOLO-World (if no manual labels)
modal run scripts/train_modal.py --action label

# 4. Train on H100
modal run scripts/train_modal.py --action train --epochs 50

# 5. Download model
modal run scripts/train_modal.py --action download
```

## Configuration

### Counting Line

Edit `src/counter.py` to adjust the counting line position:

```python
line_start = (width - 50, 100)   # Top-right
line_end = (200, height - 50)    # Bottom-left
```

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `conf` | 0.25 | Detection confidence threshold |
| `tracker` | bytetrack.yaml | Tracking algorithm |
| `epochs` | 50 | Training epochs |
| `imgsz` | 640 | Input image size |

## Tech Stack

| Component | Technology |
|-----------|------------|
| Detection | YOLOv11 (Ultralytics) |
| Tracking | ByteTrack |
| Auto-labeling | YOLO-World |
| Video Processing | OpenCV |
| Cloud Training | Modal (H100 GPU) |
| Package Manager | uv |

## License

MIT
