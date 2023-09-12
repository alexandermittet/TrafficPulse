import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd


def load_model():
    """Load the YOLOv5 model from ultralytics repository."""
    return torch.hub.load("ultralytics/yolov5", "yolov5s")


def perform_inference(model, image_path):
    """Perform inference on an image using the given YOLOv5 model.

    Args:
        model: The YOLOv5 model.
        image_path: Path to the image file.

    Returns:
        The inference results.
    """
    return model(image_path)


def display_and_save_results(results, image_path):
    """Display and save the inference results on the original image.

    Args:
        results: The inference results.
        image_path: Path to the image file.
    """
    # Read and convert the image to RGB format
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a Matplotlib figure
    plt.figure()

    # Display the image
    plt.imshow(image_rgb)

    # Extract bounding box coordinates from the results
    res1 = pd.DataFrame(results.pandas().xyxy[0])

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
    plt.savefig("data/TestImgWithBBox.png")


if __name__ == "__main__":
    model = load_model()
    image_path = "/Users/marcusnsr/Desktop/AoM/Data/Testimg.jpg"
    results = perform_inference(model, image_path)
    results.print()
    display_and_save_results(results, image_path)
