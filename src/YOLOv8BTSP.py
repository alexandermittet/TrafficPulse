# YOLOv8BTSP.py

# Importing functions and constants:
from functions import *
from constants import *

# Settings
LINE_START = Point(0, 400)
LINE_END = Point(1280, 400)


# Using a dataclass to represent arguments for BYTETracker
@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25  # Threshold for tracking
    track_buffer: int = 30  # Buffer for tracking
    match_thresh: float = 0.8  # Matching threshold
    aspect_ratio_thresh: float = 3.0  # Aspect ratio threshold
    min_box_area: float = 1.0  # Minimum bounding box area
    mot20: bool = False  # MOT20 mode flag


def main():
    # Initialize required components for processing the video
    (
        model,
        tracker,
        cap,
        info,
        generator,
        line_counter,
        box_annotator,
        line_annotator,
        CLASS_NAMES_DICT,
    ) = initialize_components(
        USE_WEBCAM, VIDEO_PATH, MODEL, BYTETrackerArgs, LINE_START, LINE_END
    )

    # VideoSink helps in writing the processed frames to a new video
    with VideoSink(get_next_video_path(video_name=TARGET_VIDEO_NAME), info) as sink:
        # Process each frame from the generator (typically frames from a video or webcam feed)
        for frame in tqdm(generator, total=info.total_frames):
            frame = process_frame(
                frame,
                model,
                tracker,
                CLASS_NAMES_DICT,
                CLASS_ID,
                line_counter,
                box_annotator,
                line_annotator,
            )
            # Write the processed frame to the output video
            sink.write_frame(frame)

            # Display the processed frame in a window
            cv2.imshow("Processed Frame", frame)

            # Exit the loop if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # If using webcam, release the camera resource
    if USE_WEBCAM:
        cap.release()

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()


# Entry point of the script
if __name__ == "__main__":
    main()
