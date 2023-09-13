import torch
import cv2
import pandas as pd


def load_model():
    """Load the YOLOv5 model from ultralytics repository."""
    return torch.hub.load("ultralytics/yolov5", "yolov5n")


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
    res1 = pd.DataFrame(results.pandas().xyxy[0])
    for index, row in res1.iterrows():
        xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
        label = row["name"]

        # Draw rectangle around the object
        cv2.rectangle(
            frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2
        )

        # Draw label text
        cv2.putText(
            frame,
            f"{label}",
            (int(xmin), int(ymin - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )


if __name__ == "__main__":
    model = load_model()

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

        # Perform inference on the current frame
        results = perform_inference(model, frame)

        # Draw bounding boxes and labels on the frame
        draw_results(frame, results)

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
