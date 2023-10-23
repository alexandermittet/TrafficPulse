# use opencv to do the job
import cv2

print(cv2.__version__)  # my version is 3.1.0
import os
import imageio
from tqdm import tqdm

# Create a folder to store extracted images
folder = "Data/custom_frames"
if not os.path.exists(folder):
    os.mkdir(folder)

# Read the video using imageio
video_path = "/Users/alexandermittet/GD_alexandermittet/uni_life/PIP/AoM/Data/custom_bench_eu_720p.mp4"


vidcap = cv2.VideoCapture(video_path)
count = 0
while True:
    success, image = vidcap.read()
    if not success:
        break
    cv2.imwrite(
        os.path.join(folder, "frame{:d}.jpg".format(count)), image
    )  # save frame as JPEG file
    count += 1
print("{} images are extacted in {}.".format(count, folder))
