# plots.py
"""
All functions for plotting the data from the CSV file.
"""

# Importing libraries:
import csv
import ast
import matplotlib.pyplot as plt
import numpy as np


def plot_multiple(filenames, interval, class_emoji_mapping):
    """
    Plot the in count and out count for each frame from multiple CSV files.

    Parameters:
    - filenames (list): A list of paths to the CSV files containing the in and out counts for each frame.
    """
    for filename in filenames:
        plot(filename)
        plot_interval(filename, interval, class_emoji_mapping)


def plot(filename):
    """
    Plot the in count and out count for each frame from the CSV file.

    Parameters:
    - filename (str): The path to the CSV file containing the in and out counts for each frame.

    Notes:
    - The CSV file should have the following format:
        ID,In Count,Out Count
        1,"{class_id: count, ...}","{class_id: count, ...}"
        2,"{class_id: count, ...}","{class_id: count, ...}"
        ...
    """
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
    image_path = filename.replace(".csv", "_cumulative.png")
    plt.savefig(image_path)


def plot_interval(filename, interval, class_emoji_mapping):
    ids = []
    in_counts = {}
    out_counts = {}

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

    def group_counts_from_cumulative(counts_dict, interval):
        grouped_counts = {}
        for class_id, counts in counts_dict.items():
            interval_counts = []
            previous_sum = 0
            for i in range(0, len(counts), interval):
                current_sum = counts[min(i + interval - 1, len(counts) - 1)]
                interval_counts.append(current_sum - previous_sum)
                previous_sum = current_sum
            grouped_counts[class_id] = interval_counts
        return grouped_counts

    grouped_in_counts = group_counts_from_cumulative(in_counts, interval)
    grouped_out_counts = group_counts_from_cumulative(
        out_counts, interval
    )  # No longer inverting the values

    bar_width = 0.35
    index = np.arange(len(grouped_in_counts[next(iter(grouped_in_counts))]))

    plt.figure(figsize=(10, 6))

    max_in = 0
    max_out = 0

    for idx, (class_id, counts) in enumerate(grouped_in_counts.items()):
        plt.bar(
            index + idx * bar_width,
            counts,
            bar_width,
            label=f"In Count (Class {class_id})",
        )
        max_in = max(max_in, max(counts))

    for idx, (class_id, counts) in enumerate(grouped_out_counts.items()):
        plt.bar(
            index + idx * bar_width,
            [-count for count in counts],  # Just invert here while plotting
            bar_width,
            label=f"Out Count (Class {class_id})",
        )
        max_out = max(max_out, max(counts))

    # Adjusting y-ticks to show positive out_counts below x-axis
    max_y = max(max_in, max_out)
    step = max(1, int(max_y / 5))
    plt.yticks(
        list(range(-max_y, max_y + 1, step)),
        [str(abs(y)) for y in range(-max_y, max_y + 1, step)],
    )

    # Annotate bars with emojis and counts
    for idx, (class_id, emoji) in enumerate(class_emoji_mapping.items()):
        total_in = in_counts.get(class_id, [0])[-1]  # get the last value
        total_out = out_counts.get(class_id, [0])[-1]  # get the last value
        total_count_for_class = total_in + total_out
        plt.annotate(
            f"{emoji} : {total_count_for_class}",
            xy=(1.02, idx * 0.1),
            xycoords="axes fraction",
            fontsize=12,
            va="center",
        )

    plt.axhline(0, color="black")
    plt.title("In Count vs. Out Count over intervals")
    plt.xlabel(f"Interval (Each interval is {interval} frames)")
    plt.ylabel("Count")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    # Save the plot as an image in the same directory as the CSV file
    image_path = filename.replace(".csv", "_interval.png")
    plt.savefig(image_path, bbox_inches="tight")
