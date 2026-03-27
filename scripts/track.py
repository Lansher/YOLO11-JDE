from collections import defaultdict
from pathlib import Path
import sys

import cv2
import numpy as np
from tqdm import tqdm

"""
# Load the YOLO11 model
model = YOLO("./../models/yolo11s-jde-tbhs.pt", task="jde")

# Open the video file
video_path = "./../videos/MOT17-13.mp4"
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # Get the total number of frames

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
with tqdm(total=total_frames, desc="Processing Frames", unit=" frames") as pbar:
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO11 tracking on the frame, persisting tracks between frames
            results = model.track(
                frame,
                tracker="smiletrack.yaml",
                persist=True,
                verbose=False
            )

            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # Display the annotated frame
            cv2.imshow("YOLO11 Tracking", annotated_frame)

            # Update the progress bar
            pbar.update(1)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

"""

# Project root: YOLO11-JDE/
_ROOT = Path(__file__).resolve().parent.parent
_WEIGHTS = _ROOT / "YOLO11s_JDE-CHMOT17-64b-100e_TBHS_m075_1280px.pt"
_VIDEO = _ROOT / "video" / "Orbbec Gemini 335_CP1564100038_20260325165759.mp4"
_VIDEO_OUTPUTS = _ROOT / "video_outputs"
_VIDEO_OUTPUTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(_ROOT))  # ensure local `YOLO11-JDE/ultralytics` is imported
from ultralytics import YOLO

model = YOLO(str(_WEIGHTS), task="jde")
results = model.track(
    source=str(_VIDEO),
    tracker="jdetracker.yaml",
    save=True,
    show=False,
    persist=True,
    project=str(_VIDEO_OUTPUTS),
    name="track",
    imgsz=(608, 1088),
)
