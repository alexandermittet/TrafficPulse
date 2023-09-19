import cv2
from ultralytics import YOLO
import time
import numpy as np
import sys
import random

sys.path.append("/Users/marcusnsr/Desktop/AoM/")

# Import DeepSORT components
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker

# constants
WEBCAM = False
DOWNSCALE_FACTOR = 2
VIDEO_PATH = "/Users/marcusnsr/Desktop/AoM/data/video30s.mp4"


# Load the YOLOv8 model
def load_model():
    """
    Load YOLOv8 model from Ultralytics repository.
    """
    return YOLO("yolov8n.pt")


def perform_inference(model, frame):
    """
    Perform object detection inference on a single video frame.

    Args:
        model: Trained YOLOv5 model.
        frame: Video frame.

    Returns:
        Inference results.
    """
    results = model(frame, classes=[1, 2, 3, 5, 7])
    return results


if __name__ == "__main__":
    # Load the YOLOv8 model
    model = load_model()

    # Initialize DeepSORT parameters
    max_cosine_distance = 0.2
    nn_budget = 100
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget
    )
    tracker = Tracker(metric)

    # Initialize video capture
    if WEBCAM:
        cap = cv2.VideoCapture(0)  # Use the default webcam (usually index 0)
    else:
        cap = cv2.VideoCapture(VIDEO_PATH)

    while cap.isOpened():
        # Start the timer
        start_time = time.time()  # Start timer
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            results = perform_inference(model, frame)

            deepsort_detections = []

            # Inside your while loop
            for r in results:
                boxes = r.boxes.xyxy  # Get bounding boxes in xyxy format

                for box in boxes:
                    dummy_feature = np.random.rand(128)
                    deepsort_detections.append(Detection(box[:4], 1.0, dummy_feature))

            # Update the tracker based on the new detections
            tracker.predict()
            tracker.update(deepsort_detections)

            # Draw tracking information
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()  # Get the corrected/predicted bounding box
                id_num = str(track.track_id)  # Get the ID for the particular track
                cv2.rectangle(
                    frame,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    (255, 0, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    id_num,
                    (int(bbox[0]), int(bbox[1])),
                    0,
                    5e-3 * 200,
                    (0, 255, 0),
                    2,
                )

            # Display the annotated frame
            cv2.imshow("YOLOv8 + DeepSORT Tracking", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break
        # End the timer and calculate fps
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print(f"FPS: {fps:.2f}")

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
