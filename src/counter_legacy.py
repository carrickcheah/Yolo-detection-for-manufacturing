"""
Conveyor Counter - Line across the conveyor belts
Counts items passing through the line - IMMEDIATE counting
"""
import cv2
import numpy as np


def main():
    video_path = "/Users/carrickcheah/Project/root_ai/box_counter/1127.mp4"

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Cannot open video"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Video: {w}x{h} @ {fps}fps")

    # Counting line - DIAGONAL across the conveyor
    line_start = (550, 150)
    line_end = (100, 500)

    # Background subtractor
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=80,
        detectShadows=False
    )

    # Detection params
    min_area = 3000
    max_area = 100000

    # Tracking
    total_count = 0
    tracks = {}  # id -> {'pos': (x,y), 'side': value, 'counted': bool}
    next_id = 0

    # Output
    out = cv2.VideoWriter(
        "counted_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    def point_side_of_line(px, py):
        """Returns positive if point is on right side of line, negative if left"""
        x1, y1 = line_start
        x2, y2 = line_end
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    print("Counting items crossing diagonal line...")

    frame_num = 0
    flash_frames = 0  # For visual feedback

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        annotated = frame.copy()
        just_counted = False

        # Background subtraction
        mask = bg_sub.apply(frame, learningRate=0.01)

        # Cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Current detections
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            cx, cy = x + bw // 2, y + bh // 2
            detections.append({'cx': cx, 'cy': cy, 'box': (x, y, bw, bh)})

        # Match detections to tracks
        matched = set()
        for det in detections:
            best_id = None
            best_dist = 80

            for tid, track in tracks.items():
                if tid in matched:
                    continue
                dist = np.sqrt((det['cx'] - track['pos'][0])**2 + (det['cy'] - track['pos'][1])**2)
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid

            current_side = point_side_of_line(det['cx'], det['cy'])

            if best_id is not None:
                # Update existing track
                prev_side = tracks[best_id]['side']
                counted = tracks[best_id]['counted']

                # IMMEDIATE crossing check - signs changed
                if not counted:
                    if (prev_side > 0 and current_side <= 0) or (prev_side < 0 and current_side >= 0):
                        total_count += 1
                        counted = True
                        just_counted = True
                        flash_frames = 5
                        print(f"\n*** ITEM #{total_count} CROSSED LINE ***")

                tracks[best_id] = {
                    'pos': (det['cx'], det['cy']),
                    'side': current_side,
                    'counted': counted,
                    'age': 0
                }
                matched.add(best_id)
            else:
                # New track
                tracks[next_id] = {
                    'pos': (det['cx'], det['cy']),
                    'side': current_side,
                    'counted': False,
                    'age': 0
                }
                next_id += 1

        # Age out old tracks
        to_del = []
        for tid in tracks:
            if tid not in matched:
                tracks[tid]['age'] += 1
                if tracks[tid]['age'] > 15:
                    to_del.append(tid)
        for tid in to_del:
            del tracks[tid]

        # Draw detections
        for det in detections:
            x, y, bw, bh = det['box']
            cv2.rectangle(annotated, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
            cv2.circle(annotated, (det['cx'], det['cy']), 5, (0, 0, 255), -1)

        # Draw counting line - flash when counting
        if flash_frames > 0:
            line_color = (0, 255, 0)  # GREEN flash
            line_thickness = 6
            flash_frames -= 1
        else:
            line_color = (0, 255, 255)  # Yellow normal
            line_thickness = 4
        cv2.line(annotated, line_start, line_end, line_color, line_thickness)

        # Count display with flash
        if flash_frames > 0:
            cv2.rectangle(annotated, (10, 10), (220, 80), (0, 100, 0), -1)
            cv2.putText(annotated, f"COUNT: {total_count}", (20, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        else:
            cv2.rectangle(annotated, (10, 10), (200, 70), (0, 0, 0), -1)
            cv2.putText(annotated, f"COUNT: {total_count}", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

        out.write(annotated)
        cv2.imshow("Conveyor Counter", annotated)

        print(f"\rFrame {frame_num} | Tracks: {len(tracks)} | COUNT: {total_count}", end="", flush=True)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\n\n{'='*40}")
    print(f"       FINAL COUNT: {total_count}")
    print(f"{'='*40}")
    print(f"Output: counted_output.mp4")


if __name__ == "__main__":
    main()
