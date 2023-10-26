# plots.py
"""
All functions for plotting the data from the CSV file.
"""

# Importing libraries:
import csv
import ast
import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def plot_multiple(filenames, speedcsv, interval, class_emoji_mapping):
    """
    Plot the in count and out count for each frame from multiple CSV files.

    Parameters:
    - filenames (list): A list of paths to the CSV files containing the in and out counts for each frame.
    """
    for filename in filenames:
        plot(filename)
        plot_interval(filename, interval, class_emoji_mapping, live=False)
        plot_vehicle_distribution(filename, class_emoji_mapping)
        plot_net_traffic_movement(filename, class_emoji_mapping)
    plot_average_speed_over_time(speedcsv)
    plot_speed_distribution_by_class(speedcsv, class_emoji_mapping)


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


def plot_interval(file_path, interval, class_emoji_mapping, live=False):
    global global_fig, global_ax
    ids = []
    in_counts = {}
    out_counts = {}

    if live:
        plt.ion()
    else:
        plt.ioff()

    # Read data from the CSV file
    with open(file_path, "r", newline="") as f:
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
        image_path = file_path.replace(".csv", "_interval.png")
        global_fig.savefig(image_path, bbox_inches="tight")
        plt.close(global_fig)


def plot_vehicle_distribution(file_path, class_maps):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Convert the string representation of dictionaries into actual dictionaries
    data["In Count"] = data["In Count"].apply(ast.literal_eval)
    data["Out Count"] = data["Out Count"].apply(ast.literal_eval)

    # Extract the counts for the last frame
    last_frame_in_counts = data.iloc[-1]["In Count"]
    last_frame_out_counts = data.iloc[-1]["Out Count"]

    # Function to process data for pie chart
    def prepare_data_for_pie_chart(count_data):
        class_names = [class_maps[k] for k in count_data.keys()]
        counts = list(count_data.values())

        # Remove classes with zero counts
        filtered_class_names = [
            class_name for class_name, count in zip(class_names, counts) if count > 0
        ]
        filtered_counts = [count for count in counts if count > 0]

        return filtered_class_names, filtered_counts

    in_class_names, in_counts = prepare_data_for_pie_chart(last_frame_in_counts)
    out_class_names, out_counts = prepare_data_for_pie_chart(last_frame_out_counts)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    colors_in = plt.cm.Paired(range(len(in_class_names)))
    explode = [0.05] * len(in_class_names)
    ax1.pie(
        in_counts,
        labels=in_class_names,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors_in,
        explode=explode,
    )
    ax1.set_title("In Vehicle Class Distribution")

    colors_out = plt.cm.Paired(range(len(out_class_names)))
    explode = [0.05] * len(out_class_names)
    ax2.pie(
        out_counts,
        labels=out_class_names,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors_out,
        explode=explode,
    )
    ax2.set_title("Out Vehicle Class Distribution")

    plt.tight_layout()

    # Save the plot as an image in the same directory as the CSV file
    image_path = file_path.replace(".csv", "_distribution.png")
    plt.savefig(image_path, bbox_inches="tight")


def plot_net_traffic_movement(file_path, class_emoji_mapping):
    """
    Visualizes the net traffic movement over time for detected vehicle classes from a given CSV file.

    Parameters:
        filepath (str): Path to the CSV file containing traffic count data.
    """

    # Load CSV
    df = pd.read_csv(file_path)

    # Convert dictionary-like strings to dictionaries
    df["In Count"] = df["In Count"].apply(ast.literal_eval)
    df["Out Count"] = df["Out Count"].apply(ast.literal_eval)

    # Compute net movement
    for vehicle_class in class_emoji_mapping.keys():
        df[class_emoji_mapping[vehicle_class]] = df.apply(
            lambda row: row["In Count"].get(vehicle_class, 0)
            - row["Out Count"].get(vehicle_class, 0),
            axis=1,
        )

    # Determine detected vehicle classes
    detected_classes = set()
    for _, row in df.iterrows():
        detected_classes.update(row["In Count"].keys())
        detected_classes.update(row["Out Count"].keys())
    detected_class_names = [
        class_emoji_mapping[cls]
        for cls in detected_classes
        if cls in class_emoji_mapping
    ]

    # Plotting
    plt.figure(figsize=(15, 10))
    for vehicle_class in detected_class_names:
        plt.plot(df["ID"], df[vehicle_class], label=vehicle_class)
    plt.title("Net Traffic Movement Over Time")
    plt.xlabel("Time Interval (ID)")
    plt.ylabel("Net Movement")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    image_path = file_path.replace(".csv", "_netmovement.png")
    plt.savefig(image_path, bbox_inches="tight")


def plot_average_speed_over_time(file_path):
    # Helper function to split and process lines from the CSV
    def process_csv_line(line):
        frame, *data_parts = line.strip().split(",")
        all_data = ",".join(data_parts)

        # Extract vehicle data from the concatenated data string
        vehicles = all_data.split(")  ")
        processed_data = []
        for vehicle in vehicles:
            if vehicle:
                processed_data.append((frame, vehicle + ")"))
        return processed_data

    # Read the CSV file line by line and process it
    processed_lines = []
    with open(file_path, "r") as f:
        next(f)  # Skip the header line
        for line in f:
            processed_lines.extend(process_csv_line(line))

    # Convert the processed lines to a DataFrame
    df = pd.DataFrame(
        processed_lines,
        columns=["Frame Number", "Data(class_id: ((x, y, x, y), speed, direction))"],
    )

    # Extract speed data from the df
    df["Speed"] = (
        df["Data(class_id: ((x, y, x, y), speed, direction))"]
        .str.extract(r"\), (\d+\.\d+),")
        .astype(float)
    )

    # Extract timestamp (Frame Number) and speed for plotting
    speed_data = df[["Frame Number", "Speed"]].copy()
    speed_data["Frame Number"] = speed_data["Frame Number"].astype(int)

    # Calculate the rolling average speed over time (using a window size of 5 for smoothing)
    speed_data["Rolling Avg Speed"] = speed_data["Speed"].rolling(window=5).mean()

    # Plot average speed of vehicles over time
    plt.figure(figsize=(12, 7))
    plt.plot(
        speed_data["Frame Number"],
        speed_data["Rolling Avg Speed"],
        color="green",
        label="Rolling Avg Speed",
    )
    plt.scatter(
        speed_data["Frame Number"],
        speed_data["Speed"],
        color="red",
        s=10,
        label="Individual Speeds",
        alpha=0.5,
    )
    plt.title("Average Vehicle Speed Over Time")
    plt.xlabel("Frame Number")
    plt.ylabel("Speed (km/h)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(file_path.replace(".csv", "_avgspeed.png"), bbox_inches="tight")


# Mapping of class IDs to emojis
CLASS_MAPS = {
    0: "People",  # Person
    1: "Cycles",  # Bicycle
    2: "Cars",  # Car
    3: "Motorcycle",  # Motorcycle
    5: "Bus",  # Bus
    7: "Trucks",  # Truck
}


def plot_speed_distribution_by_class(file_path, class_map):
    """
    Reads the CSV file, processes the data, and plots the speed distribution by detected vehicle classes.

    Parameters:
    - file_path (str): Path to the CSV file containing speed data.
    - class_map (dict): Mapping of class IDs to class names.
    """

    # Read the CSV line by line
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Extract frame numbers and vehicle data using regex
    frame_numbers = []
    vehicle_infos = []

    pattern = r'(\d+),(".*?")'
    matches = re.findall(pattern, "".join(lines[1:]))

    for match in matches:
        frame_number = match[0]
        vehicles = match[1].split('","')
        for vehicle_info in vehicles:
            frame_numbers.append(frame_number)
            vehicle_infos.append(vehicle_info.strip('"'))

    # Convert to DataFrame
    data = pd.DataFrame({"Frame": frame_numbers, "Vehicle_Info": vehicle_infos})

    # Function to extract vehicle information
    def extract_vehicle_info(row):
        pattern = r"(\d+): \(\((\d+.\d+), (\d+.\d+), (\d+.\d+), (\d+.\d+)\), (\d+.\d+), \'(\w+)\'\)"
        matches = re.findall(pattern, row)
        vehicles = [
            {
                "class_id": int(m[0]),
                "x1": float(m[1]),
                "y1": float(m[2]),
                "x2": float(m[3]),
                "y2": float(m[4]),
                "speed": float(m[5]),
                "direction": m[6],
            }
            for m in matches
        ]
        return vehicles

    # Extract vehicle information and process data
    data["Vehicles"] = data["Vehicle_Info"].apply(extract_vehicle_info)
    data["Frame"] = data["Frame"].astype(int)
    data = data.explode("Vehicles").reset_index(drop=True)
    for col in ["class_id", "x1", "y1", "x2", "y2", "speed", "direction"]:
        data[col] = data["Vehicles"].apply(
            lambda x: x[col] if isinstance(x, dict) else None
        )
    data = data.drop(columns=["Vehicle_Info", "Vehicles"])
    data["class_name"] = data["class_id"].map(class_map)
    present_classes = data["class_name"].dropna().unique()

    # Plot the data
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=data[data["class_name"].isin(present_classes)],
        x="class_name",
        y="speed",
        hue="direction",
        order=present_classes,
    )
    plt.title("Speed Distribution by Detected Vehicle Classes")
    plt.xlabel("Vehicle Class")
    plt.ylabel("Speed (Units)")
    plt.legend(title="Direction")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(file_path.replace(".csv", "_speeddist.png"), bbox_inches="tight")
