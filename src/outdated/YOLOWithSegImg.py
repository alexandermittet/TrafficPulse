from PIL import Image
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n-seg.pt")  # load an official model

# Run inference on 'bus.jpg'
results = model("/Users/marcusnsr/Desktop/AoM/data/testimg3.jpg")  # results list

# Show the results
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save("data/output/results.jpg")  # save image
