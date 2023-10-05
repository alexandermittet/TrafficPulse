import cv2
import numpy as np

# Global variable to store points
points = []


def click_event(event, x, y, flags, params):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        if len(points) == 2:
            cv2.line(frame, points[0], points[1], (0, 255, 0), 2)
            cv2.imshow("image", frame)


# Read video and get the first frame
cap = cv2.VideoCapture("/Users/marcusnsr/Desktop/AoM/data/video30s.mp4")
ret, frame = cap.read()

if not ret:
    print("Cannot read video")
    exit()

# Set the callback function for mouse events
cv2.imshow("image", frame)
cv2.setMouseCallback("image", click_event)

# Wait for two points to be selected
cv2.waitKey(0)

# You can save points to a file or print them
print("Selected points:", points)

# Release video and close window
cap.release()
cv2.destroyAllWindows()

# Now, you can use the selected points as LINE_START and LINE_END
LINE_START = points[0]
LINE_END = points[1]

# [(367, 291), (462, 294)]
