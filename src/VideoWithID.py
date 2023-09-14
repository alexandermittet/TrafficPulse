import torch
import cv2
import pandas as pd
import time
from sort import Sort

### SWITCH TO TRUE IF USING WEBCAM, FALSE IF USING VIDEO FILE ###
WEBCAM = False  # Set this to True if you want to use a webcam

# IOU threshold for object tracking
IOU_THRESHOLD = 0.5


def load_model():
    """Load the YOLOv5 model from ultralytics repository.

    Returns:
        torch.nn.Module: The loaded YOLOv5 model.
    """
    return torch.hub.load("ultralytics/yolov5", "yolov5s")


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IOU) between two bounding boxes.

    Args:
        box1 (list): [xmin, ymin, xmax, ymax] of the first bounding box.
        box2 (list): [xmin, ymin, xmax, ymax] of the second bounding box.

    Returns:
        float: The IOU value.
    """
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # Calculate intersection area
    intersectionx1 = max(x1, x3)
    intersectiony1 = max(y1, y3)
    intersectionx2 = min(x2, x4)
    intersectiony2 = min(y2, y4)

    intersection_area = max(0, intersectionx2 - intersectionx1 + 1) * max(
        0, intersectiony2 - intersectiony1 + 1
    )

    # Calculate areas of both bounding boxes
    area_box1 = (x2 - x1 + 1) * (y2 - y1 + 1)
    area_box2 = (x4 - x3 + 1) * (y4 - y3 + 1)

    # Calculate IOU
    iou = intersection_area / (area_box1 + area_box2 - intersection_area)

    return iou


def update_tracker_with_overlap(tracker, detections):
    """Update the tracker based on the detected bounding boxes using overlap-based ID assignment.

    Args:
        tracker (Sort): The tracker object to update.
        detections (list): List of [xmin, ymin, xmax, ymax, confidence, class] bounding boxes.
    """
    trackers = tracker.update(detections)
    for track in trackers:
        xmin, ymin, xmax, ymax, track_id = (
            int(track[0]),
            int(track[1]),
            int(track[2]),
            int(track[3]),
            int(track[4]),
        )

        # Draw rectangle around the object
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        # Draw label text along with Track ID
        cv2.putText(
            frame,
            f"ID {track_id}",
            (xmin, ymin - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )


if __name__ == "__main__":
    """Main function to load the model, initialize trackers and process a video file."""
    model = load_model()
    mot_tracker = Sort()

    if WEBCAM:
        # Initialize webcam capture
        cap = cv2.VideoCapture(0)  # Use the default webcam (usually index 0)
    else:
        # Initialize video capture
        video_path = "data/video30s.mp4"
        cap = cv2.VideoCapture(video_path)

    # Initialize output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        "data/output/TestVideoWithBBox.mp4",
        fourcc,
        30.0,
        (int(cap.get(3)), int(cap.get(4))),
    )

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Stream ended.")
            break

        # Start the timer
        start_time = time.time()

        # Perform inference on the current frame
        results = model(frame)

        # Convert results to a pandas DataFrame
        cur = pd.DataFrame(results.pandas().xyxy[0])
        detections = cur[["xmin", "ymin", "xmax", "ymax", "confidence", "class"]].values

        # Update the tracker based on the detected bounding boxes
        update_tracker_with_overlap(mot_tracker, detections)

        # End the timer and calculate latency
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert time to milliseconds
        fps = 1 / (end_time - start_time)  # FPS = 1 / time to process frame
        print(f"Frame processed in {latency:.2f} ms")
        print(f"FPS: {fps:.2f}")

        # Write the frame to the output video
        out.write(frame)

        # Display the frame
        cv2.imshow("YOLOv5 on Video", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture and writer objects
    cap.release()
    out.release()

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()
