# functions.py
import os
import sys
import numpy as np

np.float = float  # Fixing numpy change (should be solved in the future)
HOME = os.getcwd()
sys.path.append(f"{HOME}/ByteTrack")

import cv2
import supervision
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


def get_next_video_path(base_path=HOME, video_name="output.mp4"):
    """
    Returns the next video path based on available run numbers.

    Args:
    - base_path (str): The base directory where the 'runs' folder exists or should be created.
    - video_name (str): The name of the output video. Defaults to 'output.mp4'.

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
    next_run_number = max(run_numbers, default=0) + 1

    # Create the run folder for the next video
    run_folder = os.path.join(runs_folder_path, "{:03}".format(next_run_number))
    os.makedirs(run_folder)

    # Construct the path to the output video within the run folder
    target_video_path = os.path.join(run_folder, video_name)

    return target_video_path
