import torch
import matplotlib.pyplot as plt
import cv2

# Model
model = torch.hub.load(
    "ultralytics/yolov5", "yolov5s"
)  # yolov5n - yolov5x6 official model
# 'custom', 'path/to/best.pt')  # custom model

# Images
im = "/Users/marcusnsr/Desktop/AoM/Data/Testimg.jpg"
# or file, Path, URL, PIL, OpenCV, numpy, list

# Inference
results = model(im)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
results.xyxy[0]  # im predictions (tensor)
print("sanity0")
print(results.pandas().xyxy[0])
print("sanity2")
results.pandas().xyxy[0]  # im predictions (pandas)
#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie

results.pandas().xyxy[0].value_counts("name")  # class counts (pandas)
# person    2
# tie       1


res1 = results.pandas().xyxy[0]


# Read the image using OpenCV
image = cv2.imread(im)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Create a Matplotlib figure
plt.figure()

# Display the image
plt.imshow(image_rgb)

# Extract bounding box coordinates from the results
for index, row in res1.iterrows():
    xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
    label = row["name"]

    # Plot rectangle around the object
    plt.gca().add_patch(
        plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
    )

    # Plot center of the bounding box
    center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
    plt.plot(center_x, center_y, marker="o", color="r", label=f"{label} center")

    # Annotate the center
    plt.text(
        center_x,
        center_y,
        f"{label}",
        fontsize=12,
        ha="right",
        va="bottom",
        color="r",
    )

# Save the plot
plt.savefig("Data/output/output.png")
