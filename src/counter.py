"""
YOLO Conveyor Box Counter - Industrial Line Crossing
Using YOLO's built-in tracker for reliable tracking
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, Tuple


class LineCounter:
    """Industrial-style line crossing counter with built-in YOLO tracking"""

    def __init__(self, line_start: Tuple[int, int], line_end: Tuple[int, int]):
        self.line_start = line_start
        self.line_end = line_end
        self.track_states: Dict[int, dict] = {}  # track_id -> {side, counted}
        self.count = 0

    def get_side_of_line(self, px: int, py: int) -> float:
        """Calculate which side of line point is on"""
        x1, y1 = self.line_start
        x2, y2 = self.line_end
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    def update(self, track_id: int, cx: int, cy: int, frame_num: int) -> bool:
        """Update track and check for line crossing. Returns True if crossed."""
        current_side = self.get_side_of_line(cx, cy)

        if track_id not in self.track_states:
            # New track
            self.track_states[track_id] = {
                'side': current_side,
                'counted': False,
                'start_pos': (cx, cy)
            }
            return False

        state = self.track_states[track_id]
        prev_side = state['side']

        # Check line crossing (either direction)
        crossed_forward = (prev_side > 0) and (current_side <= 0)
        crossed_backward = (prev_side < 0) and (current_side >= 0)
        crossed = (crossed_forward or crossed_backward) and not state['counted']

        if crossed:
            self.count += 1
            state['counted'] = True
            print(f"*** COUNT {self.count} @ frame {frame_num}: Track {track_id} crossed line ***")

        # Update state
        state['side'] = current_side
        return crossed


def main():
    from pathlib import Path

    ROOT = Path(__file__).parent.parent
    video_path = ROOT / "data" / "videos" / "input.mp4"
    model_path = ROOT / "models" / "best_box_detector.pt"

    model = YOLO(model_path)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: Cannot open video")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

    # DIAGONAL COUNTING LINE
    line_start = (width - 50, 100)
    line_end = (200, height - 50)

    print(f"Counting line: {line_start} -> {line_end}")

    counter = LineCounter(line_start, line_end)

    output_path = ROOT / "data" / "videos" / "output.mp4"
    out = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    flash_frames = 0
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        display = frame.copy()

        # Use YOLO's built-in tracking (more reliable than manual tracking)
        results = model.track(frame, persist=True, verbose=False, conf=0.25, tracker="bytetrack.yaml")

        crossed_this_frame = False

        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue

            boxes = r.boxes
            for i in range(len(boxes)):
                # Get track ID from YOLO tracker
                track_id = int(boxes.id[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Update counter
                if counter.update(track_id, cx, cy, frame_num):
                    crossed_this_frame = True

                # Draw box and centroid
                cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Color based on side of line
                side = counter.get_side_of_line(cx, cy)
                color = (255, 0, 0) if side > 0 else (0, 0, 255)  # Blue=before, Red=after
                cv2.circle(display, (cx, cy), 6, color, -1)

                # Draw track ID
                cv2.putText(display, f"ID:{track_id}", (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if crossed_this_frame:
            flash_frames = 10

        # Draw counting line
        line_color = (0, 255, 0) if flash_frames > 0 else (0, 255, 255)
        cv2.line(display, line_start, line_end, line_color, 3)
        flash_frames = max(0, flash_frames - 1)

        # Draw count
        cv2.rectangle(display, (10, 10), (250, 80), (0, 0, 0), -1)
        cv2.putText(display, f"COUNT: {counter.count}", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.putText(display, f"Frame: {frame_num}", (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(display)
        cv2.imshow("Box Counter", display)

        if frame_num % 100 == 0:
            print(f"Frame {frame_num}/{total_frames} | Tracks: {len(counter.track_states)} | Count: {counter.count}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Summary
    print(f"\n--- SUMMARY ---")
    counted = sum(1 for s in counter.track_states.values() if s['counted'])
    not_counted = len(counter.track_states) - counted
    print(f"  Total tracks: {len(counter.track_states)}")
    print(f"  Counted: {counted}")
    print(f"  Not counted: {not_counted}")

    print(f"\n{'='*50}")
    print(f"  FINAL COUNT: {counter.count}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
