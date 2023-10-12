# plots.py
"""
All functions for plotting the data from the CSV file.
"""

from functions import *


def plot_multiple(filenames, interval):
    """
    Plot the in count and out count for each frame from multiple CSV files.

    Parameters:
    - filenames (list): A list of paths to the CSV files containing the in and out counts for each frame.
    """
    for filename in filenames:
        plot(filename)
        plot_interval(filename, interval)


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


def plot_interval(filename, interval):
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
    grouped_out_counts = {
        class_id: [-count for count in counts]
        for class_id, counts in group_counts_from_cumulative(
            out_counts, interval
        ).items()
    }

    bar_width = 0.35
    index = np.arange(len(grouped_in_counts[next(iter(grouped_in_counts))]))

    plt.figure(figsize=(10, 6))

    for idx, (class_id, counts) in enumerate(grouped_in_counts.items()):
        plt.bar(
            index + idx * bar_width,
            counts,
            bar_width,
            label=f"In Count (Class {class_id})",
        )

    for idx, (class_id, counts) in enumerate(grouped_out_counts.items()):
        plt.bar(
            index + idx * bar_width,
            counts,
            bar_width,
            label=f"Out Count (Class {class_id})",
            bottom=0,
        )

    plt.axhline(0, color="black")
    plt.title("In Count vs. Out Count over intervals")
    plt.xlabel(f"Interval (Each interval is {interval} frames)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    # Save the plot as an image in the same directory as the CSV file
    image_path = filename.replace(".csv", "_interval.png")
    plt.savefig(image_path)
