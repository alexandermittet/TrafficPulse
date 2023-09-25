# YOLOv8BTSP.py

# Importing functions and constants:
from functions import *
from constants import *

# Settings
LINE_START = Point(0, 400)
LINE_END = Point(1280, 400)


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


if __name__ == "__main__":
    # Init YOLO model
    model = YOLO(MODEL)
    model.fuse()

    # Get class names
    CLASS_NAMES_DICT = model.model.names

    # Initatlize ByteTracker for tracking
    tracker = BYTETracker(BYTETrackerArgs())

    # Get VideoInfo from our source:
    info = VideoInfo.from_video_path(VIDEO_PATH)

    # Get frames from video
    generator = get_video_frames_generator(VIDEO_PATH)

    # Create LineCounter instance
    line_counter = LineCounter(start=LINE_START, end=LINE_END)

    # Init BoxAnnotator
    box_annotator = BoxAnnotator(
        color=ColorPalette(), thickness=2, text_thickness=2, text_scale=1
    )
    line_annotator = LineCounterAnnotator(thickness=2, text_thickness=2, text_scale=1)

    # Main program
    with VideoSink(get_next_video_path(video_name=TARGET_VIDEO_NAME), info) as sink:
        # Loop through the video frames
        for frame in tqdm(generator, total=info.total_frames):
            # Model prediction on current frame
            results = model(frame)
            detections = Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int),
            )

            # Removeing the unwanted classes
            mask = np.array(
                [class_id in CLASS_ID for class_id in detections.class_id], dtype=bool
            )
            detections.filter(mask=mask, inplace=True)

            # Track the detection
            tracks = tracker.update(
                output_results=boxformatting(detections=detections),
                img_info=frame.shape,
                img_size=frame.shape,
            )
            tracker_id = dectect_track_matcher(detections=detections, tracks=tracks)
            detections.tracker_id = np.array(tracker_id)

            # Removing detections without trackers
            mask = np.array(
                [tracker_id is not None for tracker_id in detections.tracker_id],
                dtype=bool,
            )
            detections.filter(mask=mask, inplace=True)

            for i in range(len(detections.xyxy)):
                # Calculate and show area:
                x1, y1, x2, y2 = map(int, detections.xyxy[i])
                area = (x2 - x1) * (y2 - y1)
                cv2.putText(
                    frame,
                    f"Area: {area}",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

                # If area > 4000 pixels^2, draw a rectangle inside the bounding box
                padding = 6
                if area > 4000:
                    cv2.rectangle(
                        frame,
                        (x1 + padding, y1 + padding),
                        (x2 - padding, y2 - padding),
                        (0, 255, 0),
                        4,
                    )  # Green rectangle
                else:
                    cv2.rectangle(
                        frame,
                        (x1 + padding, y1 + padding),
                        (x2 - padding, y2 - padding),
                        (255, 0, 0),
                        4,
                    )  # Green rectanglex

            # Formatting labels
            labels = [
                f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id in detections
            ]

            # Updating line counter
            line_counter.update(detections=detections)
            # Annotate and display frame
            frame = box_annotator.annotate(
                frame=frame, detections=detections, labels=labels
            )
            line_annotator.annotate(frame=frame, line_counter=line_counter)
            sink.write_frame(frame)
