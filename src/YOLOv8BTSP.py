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


def main():
    (
        model,
        tracker,
        cap,
        info,
        generator,
        line_counter,
        box_annotator,
        line_annotator,
        CLASS_NAMES_DICT,
    ) = initialize_components(
        USE_WEBCAM, VIDEO_PATH, MODEL, BYTETrackerArgs, LINE_START, LINE_END
    )

    with VideoSink(get_next_video_path(video_name=TARGET_VIDEO_NAME), info) as sink:
        for frame in tqdm(generator, total=info.total_frames):
            frame = process_frame(
                frame,
                model,
                tracker,
                CLASS_NAMES_DICT,
                CLASS_ID,
                line_counter,
                box_annotator,
                line_annotator,
            )
            sink.write_frame(frame)
            cv2.imshow("Processed Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if USE_WEBCAM:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
