from functions import *
from constants import *

import cv2
import os
import glob


def get_image_frames_generator(img_paths):
    """
    Yields frames from a list of image paths.
    """
    for img_path in img_paths:
        yield cv2.imread(img_path)


def initialize_components_BM(IMG_DIR_PATH, MODEL, BYTETrackerArgs):
    """
    Initializes components for object detection and tracking, including setting up the video source,
    loading the YOLO model, and setting up annotations and line counters.

    Parameters:
    - USE_WEBCAM (bool): Flag to use webcam as video source. If set to False, uses the video path provided.
    - VIDEO_PATH (str): Path to the video file to be processed. Only used if USE_WEBCAM is False.
    - MODEL (str): Path to the YOLO model file.
    - BYTETrackerArgs (class): Class or function to generate arguments for BYTETracker initialization.
    - LINE_START (tuple): Starting point (x, y) of the line used in line counting.
    - LINE_END (tuple): Ending point (x, y) of the line used in line counting.

    Returns:
    - model (YOLO): Initialized YOLO model.
    - tracker (BYTETracker): Initialized BYTETracker object.
    - cap (cv2.VideoCapture): Initialized video capture object.
    - info (VideoInfo): Information about the video source.
    - generator (generator): Generator to fetch video frames.
    - line_counter (LineCounter): Initialized line counter object.
    - box_annotator (BoxAnnotator): Initialized box annotator for drawing bounding boxes.
    - line_annotator (LineCounterAnnotator): Initialized line annotator for visualizing the counting line.
    - CLASS_NAMES_DICT (dict): Dictionary of class names used by the YOLO model.

    Note:
    Assumes that the YOLO, BYTETracker, VideoInfo, webcam_generator, get_video_frames_generator, LineCounter,
    BoxAnnotator, LineCounterAnnotator, and ColorPalette classes/functions are imported and available in the same module.
    """

    model = YOLO(MODEL)
    model.fuse()
    CLASS_NAMES_DICT = model.model.names
    tracker = BYTETracker(BYTETrackerArgs())

    # Get all image paths in the directory
    img_paths = sorted(glob.glob(os.path.join(IMG_DIR_PATH, "*")))
    # Set the generator to read images from the directory
    generator = get_image_frames_generator(img_paths)
    # Set dummy video info for the images (since we don't have FPS or duration)
    info = VideoInfo(fps=30, width=1280, height=720)  # Use your actual image dimensions

    box_annotator = BoxAnnotator(
        color=ColorPalette(),
        thickness=2,
        text_thickness=1,
        text_scale=0.5,
        text_padding=3,
    )

    return (
        model,
        tracker,
        info,
        generator,
        box_annotator,
        CLASS_NAMES_DICT,
    )


def process_frame_BM(
    frame,
    model,
    tracker,
    CLASS_NAMES_DICT,
    CLASS_ID,
    box_annotator,
):
    """
    Process a given frame to detect and annotate objects based on the provided model and parameters.

    Parameters:
    - frame (numpy.ndarray): The image frame to be processed.
    - model: The pre-trained model used for object detection.
    - tracker: The object tracker instance.
    - CLASS_NAMES_DICT (dict): A dictionary mapping class IDs to their respective class names.
    - CLASS_ID (list): List of class IDs of interest for filtering detections.
    - line_counter: An instance responsible for counting objects crossing a predefined line.
    - box_annotator: An instance responsible for annotating bounding boxes on the frame.
    - line_annotator: An instance responsible for annotating lines on the frame.

    Returns:
    - numpy.ndarray: The annotated frame.

    The function carries out the following tasks:
    1. Use the model to detect objects in the frame.
    2. Convert detection results to a standard format.
    3. Filter out detections based on the provided class IDs.
    4. Update the tracker with the filtered detections.
    5. Match the detections with tracker IDs.
    6. Filter detections again based on valid tracker IDs.
    7. Annotate the frame with the area of detected objects and a bounding box.
    8. Generate labels for detections using class names, confidence, and tracker IDs.
    9. Update the line counter based on detections.
    10. Annotate the frame with the generated labels and lines.
    """
    # Get the detection results from the model for the given frame
    results = model(frame)

    # Extract the relevant detection attributes from the results and store in Detections object
    detections = Detections(
        xyxy=results[0].boxes.xyxy.cpu().numpy(),  # Bounding box coordinates
        confidence=results[0]
        .boxes.conf.cpu()
        .numpy(),  # Confidence scores of detections
        class_id=results[0]
        .boxes.cls.cpu()
        .numpy()
        .astype(int),  # Class IDs of detections
    )

    # Create a mask to filter detections based on the predefined CLASS_ID list
    mask = np.array(
        [class_id in CLASS_ID for class_id in detections.class_id], dtype=bool
    )
    # Filter the detections based on the mask
    detections.filter(mask=mask, inplace=True)

    # Update the tracker with the filtered detections and obtain tracking results
    tracks = tracker.update(
        output_results=boxformatting(detections=detections),
        img_info=frame.shape,
        img_size=frame.shape,
    )

    # Match the detections with the corresponding tracker IDs
    tracker_id = dectect_track_matcher(detections=detections, tracks=tracks)
    detections.tracker_id = np.array(tracker_id)

    # Create a mask to filter out detections without any associated tracker ID
    mask = np.array(
        [tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool
    )
    detections.filter(mask=mask, inplace=True)

    # For each detection, annotate the frame with the bounding box and the area
    for i in range(len(detections.xyxy)):
        x1, y1, x2, y2 = map(int, detections.xyxy[i])
        area = (x2 - x1) * (y2 - y1)
        # Display the area of the detection on the frame
        cv2.putText(
            frame,
            f"Area: {area}",
            (x1, y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )
        padding = 6
        # Set the color of the bounding box based on the area of the detection
        if area > 65000:
            color = (0, 0, 255)
        elif area < 55000:
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)

        # Draw the bounding box on the frame
        cv2.rectangle(
            frame, (x1 + padding, y1 + padding), (x2 - padding, y2 - padding), color, 4
        )

    # Generate labels for each detection to display the tracker ID, class name, and confidence
    labels = [
        f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id in detections
    ]

    # Annotate the frame with the detection boxes and associated labels
    frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
    # Annotate the frame with any additional lines

    # Return the annotated frame
    return frame


def main():
    # Initialize required components for processing the video
    (
        model,
        tracker,
        info,
        generator,
        box_annotator,
        CLASS_NAMES_DICT,
    ) = initialize_components_BM(
        TEST_SET_PATH,
        MODEL,
        BYTETrackerArgs,
    )

    # VideoSink helps in writing the processed frames to a new video
    with VideoSink(
        get_next_video_path(
            video_name="result_" + TARGET_VIDEO_NAME, force_new_run=True
        ),
        info,
    ) as sink:
        # Process each frame from the generator (typically frames from a video or webcam feed)
        for frame in tqdm(generator, total=info.total_frames):
            # Save the original frame to the original video
            frame = process_frame_BM(
                frame,
                model,
                tracker,
                CLASS_NAMES_DICT,
                CLASS_ID,
                box_annotator,
            )
            # Write the processed frame to the output video
            sink.write_frame(frame)

            # Display the processed frame in a window
            cv2.imshow("Processed Frame", frame)

            # Exit the loop if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()
