# functions.py
import os
import sys
import numpy as np
from constants import *

np.float = float  # Fixing numpy change (should be solved in the future)
HOME = os.getcwd()
sys.path.append(f"{HOME}/ByteTrack")

import cv2
import csv
import ast
import supervision
import matplotlib.pyplot as plt
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from ByteTrack import yolox
from ultralytics import YOLO
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from typing import List
from tqdm import tqdm


def boxformatting(detections: Detections) -> np.ndarray:
    """
    Convert detection information into a NumPy array with horizontally stacked columns.

    This function takes a Detections object and horizontally stacks the xyxy coordinates
    and confidence scores into a single NumPy array.

    Parameters:
    -----------
    detections : Type
        A Detections object containing xyxy coordinates and confidence scores.
        Replace 'Type' with the actual type of Detections if it's custom.

    Returns:
    --------
    np.ndarray
        A NumPy array with horizontally stacked columns containing xyxy coordinates
        and confidence scores.

    Example:
    --------
    >>> d = Detections(xyxy=[[1, 2, 3, 4], [5, 6, 7, 8]], confidence=[0.9, 0.8])
    >>> boxformatting(d)
    array([[1, 2, 3, 4, 0.9],
           [5, 6, 7, 8, 0.8]])
    """

    # Convert the detections of boxes and conf to a NumPy hstack
    converted = np.hstack((detections.xyxy, detections.confidence[:, np.newaxis]))
    return converted


def trackformatting(tracks: List[STrack]) -> np.ndarray:
    """
    Convert a list of STrack objects into a NumPy array containing their bounding box coordinates.

    This function takes a list of STrack objects, extracts the top-left and bottom-right coordinates
    of each track's bounding box, and stores them in a NumPy array.

    Parameters:
    -----------
    tracks : List[Type]
        A list of STrack objects, each containing a 'tlbr' attribute that represents the
        top-left and bottom-right coordinates of the bounding box.
        Replace 'Type' with the actual type of STrack if it's custom.

    Returns:
    --------
    np.ndarray
        A NumPy array containing the top-left and bottom-right coordinates of each track's
        bounding box. The dtype of the array is float.

    Example:
    --------
    >>> t1 = STrack(tlbr=[1, 2, 3, 4])
    >>> t2 = STrack(tlbr=[5, 6, 7, 8])
    >>> trackformatting([t1, t2])
    array([[1., 2., 3., 4.],
           [5., 6., 7., 8.]])
    """
    # Initialize an empty list to store track boxes
    track_boxes = []

    # Loop through each track in the list of tracks
    for track in tracks:
        # Get the top-left and bottom-right coordinates of the track
        tlbr = track.tlbr

        # Append the coordinates to the track_boxes list
        track_boxes.append(tlbr)

    # Convert the list of track boxes to a NumPy array with dtype=float
    track_boxes_array = np.array(track_boxes, dtype=float)

    return track_boxes_array


def dectect_track_matcher(detections: Detections, tracks: List[STrack]) -> Detections:
    """
    Match detections with tracks based on Intersection over Union (IoU) and return the corresponding track IDs.

    This function takes a list of Detections objects and a list of STrack objects, calculates the IoU
    between each detection and track, and assigns the track ID to the detection with the highest IoU.

    Parameters:
    -----------
    detections : Type
        A Detections object containing xyxy coordinates and confidence scores.
        Replace 'Type' with the actual type of Detections if it's custom.

    tracks : List[Type]
        A list of STrack objects, each containing a 'tlbr' attribute that represents the
        top-left and bottom-right coordinates of the bounding box and a 'track_id' attribute.
        Replace 'Type' with the actual type of STrack if it's custom.

    Returns:
    --------
    List[Union[int, None]] or np.ndarray
        A list containing the track IDs corresponding to each detection. If no match is found,
        the ID is set to None. If there are no detections or tracks, an empty NumPy array is returned.

    Example:
    --------
    >>> d = Detections(xyxy=[[1, 2, 3, 4], [5, 6, 7, 8]], confidence=[0.9, 0.8])
    >>> t1 = STrack(tlbr=[1, 2, 3, 4], track_id=1)
    >>> t2 = STrack(tlbr=[5, 6, 7, 8], track_id=2)
    >>> dectect_track_matcher(d, [t1, t2])
    [1, 2]
    """

    # Check if there are any detections and tracks; if not, return an empty array.
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    # Format the tracks for further calculations.
    formattedtracks = trackformatting(tracks=tracks)

    # Calculate the Intersection-over-Union (IoU) between each detection and track.
    iou = box_iou_batch(formattedtracks, detections.xyxy)

    # Find the index of the track with the highest IoU for each detection.
    track4detect = np.argmax(iou, axis=1)

    # Initialize a list to store the IDs of the best-matching tracks for each detection.
    ids = [None] * len(detections)

    # Assign track IDs to detections based on highest IoU.
    for id_index, detect_index in enumerate(track4detect):
        # Check if the IoU is not zero before assigning the ID.
        if iou[id_index, detect_index] != 0:
            ids[detect_index] = tracks[id_index].track_id

    # Return the list of IDs, one for each detection.
    return ids


def initialize_components(
    USE_WEBCAM, VIDEO_PATH, MODEL, BYTETrackerArgs, LINE_START, LINE_END
):
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

    # Initialize cap with a default value
    cap = None

    # Check if we are using webcam
    if USE_WEBCAM:
        # Set to default camera
        cap = cv2.VideoCapture(WEBCAM_ID)
        # Det the video info from the default camera
        info = VideoInfo.from_video_path(WEBCAM_ID)
        # Set the generator to the webcam generator
        generator = webcam_generator(cap)
    else:
        # Set the video capture to the video path
        info = VideoInfo.from_video_path(VIDEO_PATH)
        # Set the generator to the video frames generator
        generator = get_video_frames_generator(VIDEO_PATH)

    # Init line counter and annotators
    line_counter = LineCounter(start=LINE_START, end=LINE_END)
    box_annotator = BoxAnnotator(
        color=ColorPalette(),
        thickness=2,
        text_thickness=1,
        text_scale=0.5,
        text_padding=3,
    )
    line_annotator = LineCounterAnnotator(thickness=1, text_thickness=1, text_scale=1)

    return (
        model,
        tracker,
        cap,
        info,
        generator,
        line_counter,
        box_annotator,
        line_annotator,
        CLASS_NAMES_DICT,
    )


def process_frame(
    frame,
    model,
    tracker,
    CLASS_NAMES_DICT,
    CLASS_ID,
    line_counter,
    box_annotator,
    line_annotator,
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

    # Update the line counter with the new detections
    line_counter.update(detections=detections)

    # Check if the file exists
    filename = get_next_video_path(video_name=TARGET_CSV_NAME)
    file_exists = os.path.exists(filename)

    with open(filename, "a", newline="") as f:  # Open the file in append mode
        writer = csv.writer(f)

        # If the file didn't exist before, write the headers
        if not file_exists:
            writer.writerow(["ID", "In Count", "Out Count"])

        # Get the next ID
        # If the file is empty, start with 1, otherwise get the next ID
        next_id = 1
        if file_exists:
            with open(filename, "r", newline="") as read_f:
                rows = list(csv.reader(read_f))
                last_row = rows[-1] if rows else None
                if last_row:
                    next_id = int(last_row[0]) + 1  # Increase ID by 1 from the last row

        writer.writerow(
            [next_id, str(line_counter.in_count), str(line_counter.out_count)]
        )

    # Annotate the frame with the detection boxes and associated labels
    frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
    # Annotate the frame with any additional lines
    line_annotator.annotate(frame=frame, line_counter=line_counter)

    # Return the annotated frame
    return frame


def get_next_video_path(base_path=HOME, video_name="output.mp4", force_new_run=False):
    """
    Returns the next video path based on available run numbers.

    Args:
    - base_path (str): The base directory where the 'runs' folder exists or should be created.
    - video_name (str): The name of the output video. Defaults to 'output.mp4'.
    - force_new_run (bool): If True, creates a new run folder. Otherwise, use the latest run folder.

    Returns:
    - str: The next available path for the video inside a numbered run folder.
    """
    # Define the path to the runs folder
    runs_folder_path = os.path.join(base_path, "runs")

    # Create the runs folder if it doesn't exist
    if not os.path.exists(runs_folder_path):
        os.makedirs(runs_folder_path)

    # Find the next available run number
    run_numbers = [
        int(folder)
        for folder in os.listdir(runs_folder_path)
        if folder.isdigit() and len(folder) == 3
    ]

    if force_new_run:
        next_run_number = max(run_numbers, default=0) + 1
        # Create the run folder for the next video
        run_folder = os.path.join(runs_folder_path, "{:03}".format(next_run_number))
        os.makedirs(run_folder)
    else:
        # Use the latest run folder
        current_run_number = max(run_numbers, default=1)
        run_folder = os.path.join(runs_folder_path, "{:03}".format(current_run_number))

    # Construct the path to the output video within the run folder
    target_video_path = os.path.join(run_folder, video_name)

    return target_video_path


def webcam_generator(cap):
    """
    Generate frames from a given video capture object.

    Parameters:
    - cap (cv2.VideoCapture): A video capture object, typically from OpenCV.

    Yields:
    - frame (numpy.ndarray): The next frame of the video capture.

    Notes:
    - The generator will break and stop yielding frames if the video capture fails to read a frame.
    """

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame


def plot(filename):
    # Dictionaries to store the data for each class ID
    ids = []
    in_counts = {}  # {class_id: [counts for each frame]}
    out_counts = {}  # {class_id: [counts for each frame]}

    # Read data from the CSV file
    with open(filename, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        for row in reader:
            ids.append(int(row[0]))
            in_count_dict = ast.literal_eval(row[1])
            out_count_dict = ast.literal_eval(row[2])

            for class_id, count in in_count_dict.items():
                if class_id not in in_counts:
                    in_counts[class_id] = []
                in_counts[class_id].append(count)

            for class_id, count in out_count_dict.items():
                if class_id not in out_counts:
                    out_counts[class_id] = []
                out_counts[class_id].append(count)

    # Plotting the data
    plt.figure(figsize=(10, 6))

    for class_id, counts in in_counts.items():
        plt.plot(ids, counts, label=f"In Count (Class {class_id})", marker="o")

    for class_id, counts in out_counts.items():
        plt.plot(
            ids,
            counts,
            label=f"Out Count (Class {class_id})",
            marker="o",
            linestyle="--",
        )

    plt.title("In Count vs. Out Count")
    plt.xlabel("ID")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot as an image in the same directory as the CSV file
    image_path = filename.replace(".csv", ".png")
    plt.savefig(image_path)

    plt.show()
