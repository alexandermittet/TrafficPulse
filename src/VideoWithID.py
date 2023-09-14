import torch
import cv2
import pandas as pd
import time
from sort import Sort


def load_model():
    """Load the YOLOv5 model from ultralytics repository.

    Returns:
        torch.nn.Module: The loaded YOLOv5 model.
    """
    return torch.hub.load("ultralytics/yolov5", "yolov5n")


def perform_inference(model, frame):
    """Perform inference on a frame using the given YOLOv5 model.

    Args:
        model (torch.nn.Module): The YOLOv5 model.
        frame (ndarray): The video frame to perform inference on.

    Returns:
        obj: The inference results.
    """
    return model(frame)


def draw_results(frame, results, tracker):
    """Draw the inference results on the frame.

    Args:
        frame (ndarray): The video frame on which to draw results.
        results (obj): The inference results to draw.
        tracker (Sort): The tracker object to update.
    """
    cur = pd.DataFrame(results.pandas().xyxy[0])
    name = cur["name"]
    detections = cur[["xmin", "ymin", "xmax", "ymax", "confidence", "class"]].values

    # Update the tracker based on the detected bounding boxes
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
            f"{name[0]} - ID {track_id}",
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

    # Initialize video capture and output writer
    video_path = "data/video.mp4"
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        "data/output/TestVideoWithBBox.mp4",
        fourcc,
        30.0,
        (int(cap.get(3)), int(cap.get(4))),
    )

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Stream ended.")
            break

        # Start the timer
        start_time = time.time()

        # Perform inference on the current frame
        results = perform_inference(model, frame)

        # Draw bounding boxes and labels on the frame
        draw_results(frame, results, mot_tracker)

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
