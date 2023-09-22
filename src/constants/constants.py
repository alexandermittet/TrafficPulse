# Constants for all PY scripts
import datetime
import cv2

# VIDEO SOURCE
WEBCAM = False
INPUT_PATH = "Data/video.mp4"
OUTPUT_PATH = f"Data/output_{datetime.datetime.now().strftime('%m-%d_%H:%M:%S')}"

# RUNTIME PARAMS
IOU_THRESHOLD = 0.4
DOWNSCALE_FACTOR = 2

# FONT ETC
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SIZE = 0.5
TEXT_COLOR = (0, 0, 255)
TEXT_THICKNESS = 1


if __name__ == "__main__":
    print(OUTPUT_PATH)
