import cv2
from ultralytics import YOLO
import time
from collections import defaultdict
import numpy as np

# constants
WEBCAM = False
DOWNSCALE_FACTOR = 2
VIDEO_PATH = "/Users/marcusnsr/Desktop/AoM/data/video30s.mp4"
IOU_THRESHHOLD = 0.5
MAX_HISTORY = 5


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
    results = model(frame, classes=[1, 2, 3, 5])
    return results


def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    xi1, yi1, xi2, yi2 = max(x1, x1_), max(y1, y1_), min(x2, x2_), min(y2, y2_)
    intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    union = box1_area + box2_area - intersection

    return intersection / union


def annotate_frame(frame, object_ids):
    for object_id, box in object_ids.items():
        x1, y1, x2, y2 = map(int, box)
        cv2.putText(
            frame,
            f"ID: {object_id}",
            (x1, y1 - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )


if __name__ == "__main__":
    # Initialize variables
    object_ids = {}
    next_object_id = 1
    object_history = defaultdict(list)

    # Load the YOLOv8 model
    model = load_model()

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

            current_object_ids = {}  # For storing current frame detections

            for r in results:
                boxes = r.boxes.xyxy  # Get bounding boxes in xyxy format
                for box in boxes:
                    best_iou = 0
                    best_object_id = None
                    best_prediction = None

                    for object_id, last_box in object_ids.items():
                        iou = calculate_iou(box, last_box)

                        # Predict future box based on history
                        history = object_history[object_id]
                        if len(history) > 1:
                            dx = history[-1][0] - history[-2][0]
                            dy = history[-1][1] - history[-2][1]
                            predicted_box = [
                                last_box[0] + dx,
                                last_box[1] + dy,
                                last_box[2] + dx,
                                last_box[3] + dy,
                            ]
                            prediction_iou = calculate_iou(box, predicted_box)

                            if prediction_iou > best_iou:
                                best_iou = prediction_iou
                                best_object_id = object_id
                                best_prediction = predicted_box

                        elif iou > best_iou:
                            best_iou = iou
                            best_object_id = object_id

                    if best_iou > IOU_THRESHHOLD:
                        current_object_ids[best_object_id] = box
                    else:
                        current_object_ids[next_object_id] = box
                        next_object_id += 1

                    # Update the object's history
                    if best_object_id is not None:
                        history = object_history[best_object_id]
                        history.append(
                            box[:2]
                        )  # Store only the top-left corner for simplicity
                        if len(history) > MAX_HISTORY:
                            del history[0]

            # Update the object ids
            object_ids = current_object_ids

            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            annotate_frame(annotated_frame, object_ids)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

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
