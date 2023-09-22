import torch
import cv2
import pandas as pd
import time
from collections import deque
from itertools import count

import constants.constants as const

# LOADED FROM constants.py:
# ### SWITCH TO TRUE IF USING WEBCAM, FALSE IF USING VIDEO FILE ###
# WEBCAM = False  # Set this to True if you want to use a webcam
# IOU_THRESHOLD = 0.4  # Adjust this threshold as needed
# DOWNSCALE_FACTOR = 2  # Adjust this factor as needed


def load_model():
    """Load the YOLOv5 model from ultralytics repository."""
    return torch.hub.load("ultralytics/yolov5", "yolov5n")


class TrackedObject:
    def __init__(self, bbox, label):
        self.bbox = bbox  # [xmin, ymin, xmax, ymax]
        self.label = label
        self.id = next(id_counter)

    def update_bbox(self, new_bbox):
        self.bbox = new_bbox


def calculate_iou(bbox1, bbox2):
    # Calculate the Intersection over Union (IOU) of two bounding boxes
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2

    # Calculate the intersection area
    x_intersection = max(0, min(int(x2), int(x4)) - max(int(x1), int(x3)))
    y_intersection = max(0, min(int(y2), int(y4)) - max(int(y1), int(y3)))
    intersection_area = x_intersection * y_intersection

    # Calculate the area of each bounding box
    area_bbox1 = (x2 - x1) * (y2 - y1)
    area_bbox2 = (x4 - x3) * (y4 - y3)

    # Calculate IOU
    iou = intersection_area / (area_bbox1 + area_bbox2 - intersection_area)
    return iou


# Initialize ID counter
id_counter = count(1)


def perform_inference(model, frame):
    """Perform inference on a frame using the given YOLOv5 model.

    Args:
        model: The YOLOv5 model.
        frame: The video frame.

    Returns:
        The inference results.
    """
    results = model(frame)
    return results


def draw_results(frame, results):
    """Draw the inference results on the frame.

    Args:
        frame: The video frame.
        results: The inference results.
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
            const.FONT,  # Font type
            const.FONT_SIZE,  # Font size
            const.TEXT_COLOR,  # Text color (in BGR format)
            const.TEXT_THICKNESS,  # Thickness of the text
        )


if __name__ == "__main__":
    model = load_model()
    if const.WEBCAM:
        # Initialize webcam capture
        cap = cv2.VideoCapture(0)  # Use the default webcam (usually index 0)
    else:
        # Initialize video capture
        cap = cv2.VideoCapture(const.INPUT_PATH)

    # Initialize output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        const.OUTPUT_PATH + "_tracked.mp4",
        fourcc,
        30.0,
        (int(cap.get(3)), int(cap.get(4))),
    )

    # Inside the loop
    tracked_objects = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Stream ended.")
            break

        # Downscale the frame
        frame = cv2.resize(
            frame, None, fx=1 / const.DOWNSCALE_FACTOR, fy=1 / const.DOWNSCALE_FACTOR
        )

        # Start the timer
        start_time = time.time()

        # Perform inference on the current frame
        results = perform_inference(model, frame)

        # Update tracked objects
        new_tracked_objects = []
        for index, row in pd.DataFrame(results.pandas().xyxy[0]).iterrows():
            bbox = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]
            label = row["name"]

            # Try to associate the detection with an existing track
            matched = False
            for obj in tracked_objects:
                iou = calculate_iou(obj.bbox, bbox)
                if iou >= const.IOU_THRESHOLD:
                    obj.update_bbox(bbox)
                    new_tracked_objects.append(obj)
                    matched = True
                    break

            # If not matched, create a new track
            if not matched:
                new_obj = TrackedObject(bbox, label)
                new_tracked_objects.append(new_obj)

        # Update the list of tracked objects
        tracked_objects = new_tracked_objects

        # Draw bounding boxes and labels on the frame
        draw_results(frame, results)

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
