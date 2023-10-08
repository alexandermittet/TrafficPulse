# benchmark.py
"""
The main program that runs benchmarking and MOT metrics on a video.
"""

# Importing functions and constants:
from functions_benchmark import *
from constants import *

if __name__ == "__main__":
    main()
    # Calculate the MOT metrics
    motMetricsEnhancedCalculator(
        "/Users/marcusnsr/Desktop/AoM/transformed_duck.txt",
        "/Users/marcusnsr/Desktop/AoM/src/runs/004/YOLOv8_BB_(10-04_10:28:41).txt",
    )
