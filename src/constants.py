# constants.py
"""
Constants used in multiple files.
"""
import os
from datetime import datetime

# Get the current date and time in a compact format
compact_datetime = datetime.now().strftime("%m-%d_%H:%M:%S")

# Toggles:
USE_WEBCAM = True  # If False, use the video file specified in VIDEO_PATH else use the webcam (set WEBCAM_ID under Constants)
GET_AREA = False  # If True, get the area of the bounding box of each object
GRID_LINES = (
    False  # If True, draw grid lines on the video (set n and m under Constants)
)
LIVE = False  # If True, show the interval plot live
GET_SPEED = True  # If True, calculate the speed of each object
# Constants:
WEBCAM_ID = 0  # The ID of the webcam to use
GRID_LINES_n = 4  # The number of vertical grid lines
GRID_LINES_m = 4  # The number of horizontal grid lines
INTERVAL = 15  # The interval to plot the in and out counts
HIST_LEN = 15  # The number of frames to plot in the histogram
PPM = 15  # The pixels per meter (calibration)
CLASS_ID = [0, 2, 3, 7]  # The class IDs to track
""" Class IDs:
0: 'person',
1: 'bicycle',
2: 'car',
3: 'motorcycle',
4: 'airplane',
5: 'bus',
6: 'train',
7: 'truck',
8: 'boat',
9: 'traffic light',
10: 'fire hydrant',
11: 'stop sign',
12: 'parking meter',
13: 'bench',
14: 'bird',
15: 'cat',
16: 'dog',
17: 'horse',
18: 'sheep',
19: 'cow',
20: 'elephant',
21: 'bear',
22: 'zebra',
23: 'giraffe',
24: 'backpack',
25: 'umbrella',
26: 'handbag',
27: 'tie',
28: 'suitcase',
29: 'frisbee',
30: 'skis',
31: 'snowboard',
32: 'sports ball',
33: 'kite',
34: 'baseball bat',
35: 'baseball glove',
36: 'skateboard',
37: 'surfboard',
38: 'tennis racket',
39: 'bottle',
40: 'wine glass',
41: 'cup',
42: 'fork',
43: 'knife',
44: 'spoon',
45: 'bowl',
46: 'banana',
47: 'apple',
48: 'sandwich',
49: 'orange',
50: 'broccoli',
51: 'carrot',
52: 'hot dog',
53: 'pizza',
54: 'donut',
55: 'cake',
56: 'chair',
57: 'couch',
58: 'potted plant',
59: 'bed',
60: 'dining table',
61: 'toilet',
62: 'tv',
63: 'laptop',
64: 'mouse',
65: 'remote',
66: 'keyboard',
67: 'cell phone',
68: 'microwave',
69: 'oven',
70: 'toaster',
71: 'sink',
72: 'refrigerator',
73: 'book',
74: 'clock',
75: 'vase',
76: 'scissors',
77: 'teddy bear',
78: 'hair drier',
79: 'toothbrush'
"""

# Paths:
SRC_DIR = os.path.dirname(os.path.abspath(__file__))  # The directory of this file
PROJECT_ROOT_DIR = os.path.join(SRC_DIR, os.pardir)  # The root directory of the project
VIDEO_PATH = os.path.join(
    PROJECT_ROOT_DIR, "data", "highway3.mp4"
)  # The path to the video file
TEST_SET_PATH = os.path.join(
    PROJECT_ROOT_DIR, "data", "MOT16-13", "img1"
)  # The path to the test set
TARGET_VIDEO_NAME = f"YOLOv8_({compact_datetime}).mp4"  # The name of the output video
TARGET_CSV_NAME = f"YOLOv8_({compact_datetime}).csv"  # The name of the output CSV file
TARGET_BBOX_NAME = (
    f"YOLOv8_BB_({compact_datetime}).txt"  # The name of the output bounding box file
)
MODEL = os.path.join(
    PROJECT_ROOT_DIR, "models", "yolov8n.pt"
)  # The path to the model / model name
""" # Models:
yolov8n.pt
yolov8s.pt
yolov8m.pt
yolov8l.pt
yolov8x.pt
"""

# Mapping of class IDs to emojis
CLASS_MAPS = {
    0: "People",  # Person
    1: "Cycles",  # Bicycle
    2: "Cars",  # Car
    3: "Motorcycle",  # Motorcycle
    5: "Bus",  # Bus
    7: "Trucks",  # Truck
}
