# functions_benchmark.py
"""
All the functions required for the benchmarking of the model.
"""

from functions import *
from constants import *

import glob
import motmetrics as mm


def get_image_frames_generator(img_paths):
    """
    Yields frames from a list of image paths.
    """
    for img_path in img_paths:
        yield cv2.imread(img_path)


def initialize_components_BM(IMG_DIR_PATH, MODEL, BYTETrackerArgs):
    """
    Initializes the required components for processing the video.

    Args:
        IMG_DIR_PATH (str): Path to the directory containing the images.
        MODEL (str): Path to the model.
        BYTETrackerArgs (class): Class containing the arguments for the tracker.

    Returns:
        model (YOLO): The YOLO model.
        tracker (BYTETracker): The tracker.
        info (VideoInfo): The video info.
        generator (generator): The generator for the frames.
        box_annotator (BoxAnnotator): The box annotator.
        CLASS_NAMES_DICT (dict): The dictionary containing the class names.
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
    Processes a single frame.

    Args:
        frame (np.ndarray): The frame to process.
        model (YOLO): The YOLO model.
        tracker (BYTETracker): The tracker.
        CLASS_NAMES_DICT (dict): The dictionary containing the class names.
        CLASS_ID (list): The list of class IDs to track.
        box_annotator (BoxAnnotator): The box annotator.

    Returns:
        frame (np.ndarray): The processed frame.
    """
    # Get the detection results from the model for the given frame
    results = model.predict(frame, device=DEVICE)

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

    # Getting some stuff for the benchmark
    frame_id = tracker.frame_id
    xywh = results[0].boxes.xywh.cpu().numpy()

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

    # Initialize empty list to store your predictions
    predictions = []

    # Loop over all detections in the frame
    for i, track_id in enumerate(tracker_id):
        # Skip None values
        if track_id is None:
            continue

        # Create a single prediction record
        prediction = {
            "FrameId": frame_id,
            "Id": track_id,
            "X": (xywh[i][0] - (xywh[i][2] / 2)),
            "Y": (xywh[i][1] - (xywh[i][3] / 2)),
            "Width": xywh[i][2],
            "Height": xywh[i][3],
            "Confidence": detections.confidence[i]
            if i < len(detections.confidence)
            else 0,
            "ClassId": detections.class_id[i] if i < len(detections.class_id) else -1,
            "Filler1": -1,
            "Filler2": -1,
            "Filler3": -1,
        }

        # Append to predictions list
        predictions.append(prediction)

    filename = get_next_video_path(video_name=TARGET_BBOX_NAME)

    if predictions:
        keys = predictions[0].keys()
        if predictions:  # Check if the list is not empty
            keys = predictions[0].keys()
            with open(filename, "a+", newline="") as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=keys)
                dict_writer.writerows(predictions)
    # Return the annotated frame
    return frame


def motMetricsEnhancedCalculator(gtSource, tSource):
    """
    Calculates the MOT metrics for the given ground truth and tracking output files.

    Args:
        gtSource (str): Path to the ground truth file.
        tSource (str): Path to the tracking output file.
    """
    # Load ground truth
    gt = np.loadtxt(gtSource, delimiter=",")

    # Load tracking output
    t = np.loadtxt(tSource, delimiter=",")

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    # Max frame number maybe different for gt and t files
    for frame in range(int(gt[:, 0].max())):
        frame += 1  # detection and frame numbers begin at 1

        # Select id, x, y, width, height for current frame
        # Required format for distance calculation is X, Y, Width, Height \
        # We already have this format
        gt_dets = gt[gt[:, 0] == frame, 1:6]  # Select all detections in gt
        t_dets = t[t[:, 0] == frame, 1:6]  # Select all detections in t

        C = mm.distances.iou_matrix(
            gt_dets[:, 1:], t_dets[:, 1:], max_iou=0.5
        )  # Format: gt, t

        # Call update once for per frame.
        # Format: gt object ids, t object ids, distance
        acc.update(
            gt_dets[:, 0].astype("int").tolist(), t_dets[:, 0].astype("int").tolist(), C
        )

    mh = mm.metrics.create()

    summary = mh.compute(
        acc,
        metrics=[
            "num_frames",
            "idf1",
            "idp",
            "idr",
            "recall",
            "precision",
            "num_objects",
            "mostly_tracked",
            "partially_tracked",
            "mostly_lost",
            "num_false_positives",
            "num_misses",
            "num_switches",
            "num_fragmentations",
            "mota",
            "motp",
        ],
        name="acc",
    )

    strsummary = mm.io.render_summary(
        summary,
        # Formatters={'mota' : '{:.2%}'.format},
        namemap={
            "idf1": "IDF1",
            "idp": "IDP",
            "idr": "IDR",
            "recall": "Rcll",
            "precision": "Prcn",
            "num_objects": "GT",
            "mostly_tracked": "MT",
            "partially_tracked": "PT",
            "mostly_lost": "ML",
            "num_false_positives": "FP",
            "num_misses": "FN",
            "num_switches": "IDsw",
            "num_fragmentations": "FM",
            "mota": "MOTA",
            "motp": "MOTP",
        },
    )
    print(strsummary)


def main():
    """
    The main function for benchmarking.
    """
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
