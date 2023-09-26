import csv
import matplotlib.pyplot as plt

filename = "/Users/marcusnsr/Desktop/AoM/src/test.csv"

# Lists to store the data
ids = []
in_counts = []
out_counts = []

# Read data from the CSV file
with open(filename, "r", newline="") as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header row
    for row in reader:
        ids.append(int(row[0]))
        in_counts.append(int(row[1]))
        out_counts.append(int(row[2]))

# Plotting the data
plt.figure(figsize=(10, 6))

plt.plot(ids, in_counts, label="In Count", marker="o")
plt.plot(ids, out_counts, label="Out Count", marker="o")

plt.title("In Count vs. Out Count")
plt.xlabel("ID")
plt.ylabel("Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
