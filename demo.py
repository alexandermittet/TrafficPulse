import sys

sys.path.append("src")
from functions import *

# Hardcoded constants:
VID_PATH = "/Users/marcusnsr/Desktop/AoM/demo/demo_video.mp4"


def main():
    point1, point2 = select_two_points_from_video(VID_PATH)

    if point1 and point2:
        LINE_START = Point(*point1)
        LINE_END = Point(*point2)
        print("Selected points:", point1, point2)
    else:
        print("Points were not selected properly.")
        exit()

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
        False,
        VID_PATH,
        "models/yolov8n.pt",
        BYTETrackerArgs,
        LINE_START,
        LINE_END,
    )

    # Process each frame from the generator (typically frames from a video or webcam feed)
    for frame in tqdm(generator, total=info.total_frames):
        # Save the original frame to the original video
        frame = process_frame(
            frame,
            model,
            tracker,
            CLASS_NAMES_DICT,
            [2],
            line_counter,
            box_annotator,
            line_annotator,
        )

        # Display the processed frame in a window
        cv2.imshow("Processed Frame", frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    while True:
        key = cv2.waitKey(1) & 0xFF  # Wait for a key press for 1 millisecond

        # Check if the 'q' key is pressed (ASCII value of 'q' is 113)
        if key == 113:
            break
    # Destroy all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
