import torch
import cv2
import pandas as pd
import time
from collections import deque, Counter
from itertools import count
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Constants
WEBCAM = False  # Set this to True if you want to use a webcam, False for a video file
IOU_THRESHOLD = 0.3  # Intersection-over-Union threshold for object tracking
DOWNSCALE_FACTOR = 2  # Factor by which the video frame will be downscaled

# Initialize global ID counter for TrackedObjects
id_counter = count(1)


def load_model():
    """
    Load YOLOv8 model from Ultralytics repository.
    """
    return YOLO("yolov8n.pt")


class TrackedObject:
    """
    Represents an object being tracked in the video.
    """

    def __init__(self, bbox, label):
        """
        Initialize a new TrackedObject.
        Args:
            bbox: Bounding box coordinates [xmin, ymin, xmax, ymax].
            label: Label of the object.
            id_counter: Counter for assigning unique IDs to TrackedObjects.
            counted: Flag to indicate if this object has been counted.
            history: Store the history of mid-points of the bounding box.
            label_history: Store the last n labels.
        """
        self.bbox = bbox
        self.label = label
        self.id = next(id_counter)
        self.counted = False
        self.history = deque(maxlen=20)
        self.label_history = deque(maxlen=10)

    def update_bbox_and_label(self, new_bbox, new_label):
        """
        Update the bounding box and label of the object.
        """
        self.bbox = new_bbox
        self.label_history.append(new_label)

    def get_persistent_label(self):
        """
        Update the bounding box and label of the object.
        """
        return Counter(self.label_history).most_common(1)[0][0]


def calculate_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IOU) between two bounding boxes.

    Args:
        bbox1, bbox2: Lists containing coordinates [xmin, ymin, xmax, ymax].

    Returns:
        IOU value.
    """
    # Calculate intersection area
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    x_intersection = max(0, min(int(x2), int(x4)) - max(int(x1), int(x3)))
    y_intersection = max(0, min(int(y2), int(y4)) - max(int(y1), int(y3)))
    intersection_area = x_intersection * y_intersection

    # Calculate area of each bounding box
    area_bbox1 = (x2 - x1) * (y2 - y1)
    area_bbox2 = (x4 - x3) * (y4 - y3)

    # Calculate IOU
    iou = intersection_area / (area_bbox1 + area_bbox2 - intersection_area)
    return iou


def perform_inference(model, frame):
    """
    Perform object detection inference on a single video frame.

    Args:
        model: Trained YOLOv5 model.
        frame: Video frame.

    Returns:
        Inference results.
    """
    results = model(frame)
    return results


def draw_results(frame, results):
    """
    Draw bounding boxes and labels on the video frame.

    Args:
        frame: Video frame.
        results: Inference results from YOLOv5.
    """
    cur = pd.DataFrame(results.pandas().xyxy[0])
    for index, row in cur.iterrows():
        xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
        label = row["name"]
        cv2.rectangle(
            frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2
        )
        cv2.putText(
            frame,
            f"{label}",
            (int(xmin), int(ymin - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )


# Main code execution starts here
if __name__ == "__main__":
    model = load_model()  # Load the YOLOv5 model

    # Initialize video capture
    if WEBCAM:
        cap = cv2.VideoCapture(0)  # Use the default webcam (usually index 0)
    else:
        video_path = "/Users/marcusnsr/Desktop/AoM/data/video30s.mp4"
        cap = cv2.VideoCapture(video_path)

    # Original dimensions
    original_width = int(cap.get(3))
    original_height = int(cap.get(4))

    # Downscaled dimensions
    downscaled_width = int(original_width / DOWNSCALE_FACTOR)
    downscaled_height = int(original_height / DOWNSCALE_FACTOR)

    # Initialize video writer for output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Initialize VideoWriter
    out = cv2.VideoWriter(
        "data/output/TestVideoWithBBox.mp4",
        fourcc,
        30.0,
        (downscaled_width, downscaled_height),
    )

    # List to keep track of TrackedObjects
    tracked_objects = []
    up_count = 0  # Counter for objects moving up
    down_count = 0  # Counter for objects moving down

    # Main loop for processing video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Stream ended.")
            break

        # Downscale the frame to speed up processing
        frame = cv2.resize(
            frame, None, fx=1 / DOWNSCALE_FACTOR, fy=1 / DOWNSCALE_FACTOR
        )

        # Perform object detection
        start_time = time.time()  # Start timer
        results = perform_inference(model, frame)
        end_time = time.time()  # End timer

        # Update tracked objects based on detection results
        new_tracked_objects = []
        for index, row in pd.DataFrame(results.pandas().xyxy[0]).iterrows():
            bbox = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]
            label = row["name"]

            # Try to associate the detection with an existing track
            matched = False
            for obj in tracked_objects:
                iou = calculate_iou(obj.bbox, bbox)
                if iou >= IOU_THRESHOLD:
                    obj.update_bbox_and_label(bbox, label)  # Update both bbox and label
                    new_tracked_objects.append(obj)
                    matched = True
                    break

            # If not matched, create a new track
            if not matched:
                new_obj = TrackedObject(bbox, label)
                new_obj.label_history.append(label)  # Initialize label history
                new_tracked_objects.append(new_obj)

        # Update the list of tracked objects
        tracked_objects = new_tracked_objects

        # Draw bounding boxes and labels on the frame
        for obj in tracked_objects:
            label = obj.get_persistent_label()  # Use persistent label
            xmin, ymin, xmax, ymax = obj.bbox
            cv2.rectangle(
                frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2
            )
            cv2.putText(
                frame,
                f"{label}",
                (int(xmin), int(ymin - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        # Draw counting line
        cv2.line(
            frame,
            (0, frame.shape[0] // 3),
            (frame.shape[1], frame.shape[0] // 3),
            (0, 255, 0),
            2,
        )

        # Update object history
        for obj in tracked_objects:
            xmin, ymin, xmax, ymax = obj.bbox
            mid_box = int((ymin + ymax) / 2)
            obj.history.append(mid_box)

        # Implement counting logic
        mid_line = frame.shape[0] // 2
        min_distance = 10
        for obj in tracked_objects:
            if not obj.counted and len(obj.history) == 20:
                direction = obj.history[-1] - obj.history[0]
                if (
                    direction > 0
                    and obj.history[0] < mid_line
                    and obj.history[-1] > mid_line + min_distance
                ):
                    down_count += 1
                    obj.counted = True
                elif (
                    direction < 0
                    and obj.history[0] > mid_line
                    and obj.history[-1] < mid_line - min_distance
                ):
                    up_count += 1
                    obj.counted = True

        # Draw counts on the frame
        cv2.putText(
            frame,
            f"Up Count: {up_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Down Count: {down_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # Draw object IDs on the frame
        for obj in tracked_objects:
            cv2.putText(
                frame,
                f"ID: {obj.id}",
                (int(obj.bbox[0]), int(obj.bbox[1] - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        # End the timer and calculate latency
        end_time = time.time()
        latency = (end_time - start_time) * 1000
        fps = 1 / (end_time - start_time)
        print(f"Frame processed in {latency:.2f} ms")
        print(f"FPS: {fps:.2f}")

        # Write the frame to the output video
        out.write(frame)

        # Display the frame
        cv2.imshow("YOLOv5 with Object Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
