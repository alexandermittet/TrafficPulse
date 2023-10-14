# plots.py
"""
All functions for plotting the data from the CSV file.
"""

# Importing libraries:
import csv
import ast
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_multiple(filenames, interval, class_emoji_mapping):
    """
    Plot the in count and out count for each frame from multiple CSV files.

    Parameters:
    - filenames (list): A list of paths to the CSV files containing the in and out counts for each frame.
    """
    for filename in filenames:
        plot(filename)
        plot_interval(filename, interval, class_emoji_mapping, live=False)
        plot_vehicle_distribution(filename, class_emoji_mapping)


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


global_fig, global_ax = None, None


def plot_interval(filename, interval, class_emoji_mapping, live=False):
    global global_fig, global_ax
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
    grouped_out_counts = group_counts_from_cumulative(out_counts, interval)

    bar_width = 0.35
    index = np.arange(len(grouped_in_counts[next(iter(grouped_in_counts))]))

    if not global_fig or not plt.fignum_exists(global_fig.number):
        plt.ion()
        global_fig, global_ax = plt.subplots(figsize=(10, 6))
    else:
        global_ax.clear()

    max_in = 0
    max_out = 0

    for idx, (class_id, counts) in enumerate(grouped_in_counts.items()):
        global_ax.bar(
            index + idx * bar_width,
            counts,
            bar_width,
            label=f"In Count (Class {class_id})",
        )
        max_in = max(max_in, max(counts))

    for idx, (class_id, counts) in enumerate(grouped_out_counts.items()):
        global_ax.bar(
            index + idx * bar_width,
            [-count for count in counts],  # Just invert here while plotting
            bar_width,
            label=f"Out Count (Class {class_id})",
        )
        max_out = max(max_out, max(counts))

    # Adjusting y-ticks to show positive out_counts below x-axis
    max_y = max(max_in, max_out)
    step = max(1, int(max_y / 5))
    global_ax.set_yticks(list(range(-max_y, max_y + 1, step)))
    global_ax.set_yticklabels([str(abs(y)) for y in range(-max_y, max_y + 1, step)])

    # Annotate bars with emojis and counts
    for idx, (class_id, emoji) in enumerate(class_emoji_mapping.items()):
        total_in = in_counts.get(class_id, [0])[-1]  # get the last value
        total_out = out_counts.get(class_id, [0])[-1]  # get the last value
        total_count_for_class = total_in + total_out
        global_ax.annotate(
            f"{emoji} : {total_count_for_class}",
            xy=(1.02, idx * 0.1),
            xycoords="axes fraction",
            fontsize=12,
            va="center",
        )

    global_ax.axhline(0, color="black")
    global_ax.set_title("In Count (top) vs. Out Count (bottom) over intervals")
    global_ax.set_xlabel(f"Interval (Each interval is {interval} frames)")
    global_ax.set_ylabel("Count")
    global_ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    global_ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    global_fig.tight_layout()

    if live:
        plt.draw()
        plt.pause(0.01)  # Optional short pause to ensure plot gets updated
    else:
        # Save the plot as an image in the same directory as the CSV file
        image_path = filename.replace(".csv", "_interval.png")
        global_fig.savefig(image_path, bbox_inches="tight")


def plot_vehicle_distribution(file_path, CLASS_EMOJI_MAPS):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Convert the string representation of dictionaries into actual dictionaries
    data["In Count"] = data["In Count"].apply(ast.literal_eval)
    data["Out Count"] = data["Out Count"].apply(ast.literal_eval)

    # Extract the counts for the last frame
    last_frame_in_counts = data.iloc[-1]["In Count"]
    last_frame_out_counts = data.iloc[-1]["Out Count"]

    # Sum the "In" and "Out" counts for each vehicle class
    total_counts = {
        k: last_frame_in_counts.get(k, 0) + last_frame_out_counts.get(k, 0)
        for k in set(last_frame_in_counts) | set(last_frame_out_counts)
    }

    class_names = [CLASS_EMOJI_MAPS[k] for k in total_counts.keys()]
    counts = list(total_counts.values())

    # Remove classes with zero counts
    filtered_class_names = [
        class_name for class_name, count in zip(class_names, counts) if count > 0
    ]
    filtered_counts = [count for count in counts if count > 0]

    # Plotting
    plt.figure(figsize=(10, 7))
    colors = plt.cm.Paired(range(len(filtered_class_names)))
    explode = [0.05] * len(filtered_class_names)
    plt.pie(
        filtered_counts,
        labels=filtered_class_names,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        explode=explode,
    )
    plt.title("Overall Vehicle Class Distribution")
    plt.tight_layout()
    # Save the plot as an image in the same directory as the CSV file
    image_path = file_path.replace(".csv", "_distribution.png")
    plt.savefig(image_path, bbox_inches="tight")
