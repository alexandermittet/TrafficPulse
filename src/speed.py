# speed.py

from functions import *
from constants import *


def calculate_centroid(tl_x, tl_y, w, h):
    """
    Calculates the centroid of a bounding box given its top-left coordinates and dimensions.

    Parameters:
    - tl_x: x-coordinate of the top-left corner of the bounding box.
    - tl_y: y-coordinate of the top-left corner of the bounding box.
    - w: width of the bounding box.
    - h: height of the bounding box.

    Returns:
    - (mid_x, mid_y): tuple representing the centroid of the bounding box.
    """

    # Calculate mid-point along the x-axis
    mid_x = int(tl_x + w / 2)

    # Calculate mid-point along the y-axis
    mid_y = int(tl_y + h / 2)

    return mid_x, mid_y


def convert_history_to_dict(track_history):
    """
    Converts tracking history to a dictionary representation. The dictionary maps object IDs
    to their historical centroids.

    Parameters:
    - track_history: List containing historical tracking data. Each entry consists of
                    object IDs, bounding box representations (tlwh), and potentially other data.

    Returns:
    - history_dict: Dictionary mapping object IDs to their historical centroids.
    """

    history_dict = {}

    # Iterate over the tracking data for each frame
    for frame_content in track_history:
        obj_ids, tlwhs, _ = frame_content

        # Iterate over each tracked object in the current frame
        for obj_id, tlwh in zip(obj_ids, tlwhs):
            # Extract bounding box parameters: top-left x, top-left y, width, height
            tl_x, tl_y, w, h = tlwh

            # Calculate the centroid for the current bounding box
            mid_x, mid_y = calculate_centroid(tl_x, tl_y, w, h)

            # Update the history dictionary with the centroid data
            if obj_id not in history_dict.keys():
                history_dict[obj_id] = [[mid_x, mid_y]]
            else:
                history_dict[obj_id].append([mid_x, mid_y])

    return history_dict


def update_and_draw(detections, frame, history_dict, history_len):
    """
    Updates the history of detections and draws lines on a frame.

    Parameters:
    - detections: The detected objects containing xyxy and tracker_id.
    - frame: The frame where lines will be drawn.
    - history_dict: A dictionary to store the history of centroids.
    - history_len: The length of the history to keep.

    Returns:
    - frame: The frame with lines drawn on it.
    - history_dict: Updated history dictionary.
    """

    # Update the history
    for detection, tracker_id in zip(detections.xyxy, detections.tracker_id):
        x1, y1, x2, y2 = map(int, detection)  # Ensure integer values
        centroid = (
            (x1 + x2) // 2,
            (y1 + y2) // 2,
        )  # Calculate the centroid of the bounding box
        if tracker_id not in history_dict:
            history_dict[tracker_id] = [(centroid, time.time())]
        else:
            history_dict[tracker_id].append((centroid, time.time()))

            if len(history_dict[tracker_id]) > history_len:
                history_dict[tracker_id].pop(0)
    # Get the set of tracker IDs currently in frame
    current_tracker_ids = set(detections.tracker_id)

    # Get the set of tracker IDs in the history
    history_tracker_ids = set(history_dict.keys())

    # Identify which tracker IDs are missing from the current frame
    missing_ids = history_tracker_ids - current_tracker_ids

    # Remove these IDs from the history_dict
    for tracker_id in missing_ids:
        del history_dict[tracker_id]

    # Draw the lines
    for tracker_id in detections.tracker_id:
        if tracker_id in history_dict and len(history_dict[tracker_id]) > 1:
            for i in range(len(history_dict[tracker_id]) - 1):
                start_point = tuple(
                    map(int, history_dict[tracker_id][i][0])
                )  # Ensure integer values and get only the coordinates
                end_point = tuple(
                    map(int, history_dict[tracker_id][i + 1][0])
                )  # Ensure integer values and get only the coordinates
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

    return frame, history_dict


def estimate_speed(location1, location2, ppm, time_elapsed):
    """
    Estimate the speed of a moving object between two locations.

    :param location1: tuple, the first location (x1, y1)
    :param location2: tuple, the second location (x2, y2)
    :param ppm: float, pixels per meter (calibration value)
    :param time_elapsed: float, time elapsed between two locations in seconds
    :return: float, speed in km/h
    """
    # Compute the distance in pixels
    d_pixel = math.sqrt(
        math.pow(location2[0] - location1[0], 2)
        + math.pow(location2[1] - location1[1], 2)
    )
    d_meters = d_pixel / ppm
    speed = d_meters / time_elapsed  # speed in meters per second
    speed_kmh = speed * 3.6  # convert to km/h
    return speed_kmh


def moving_average_speed_from_history(history, ppm, window_size=3):
    # Ensure there are enough positions in the history
    if (
        len(history) < window_size + 1
    ):  # +1 because we need both start and end for each interval
        return 0

    speeds = []

    # Calculate speed for each frame and take the average of the last few
    for i in range(-window_size, 0):
        pos1, time1 = history[i]
        pos2, time2 = history[i + 1]

        # Ensure pos1 is the earlier position
        if time1 > time2:
            pos1, pos2 = pos2, pos1
            time1, time2 = time2, time1

        # Calculate speed
        speed_kmh = estimate_speed(pos1, pos2, ppm, time2 - time1)
        speeds.append(speed_kmh)

    return sum(speeds) / len(speeds)
