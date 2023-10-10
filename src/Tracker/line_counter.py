import cv2
import numpy as np

from supervision.draw.color import Color
from supervision.geometry.dataclasses import Point, Rect, Vector
from supervision.tools.detections import Detections


class LineCounter:
    def __init__(self, start: Point, end: Point):
        """
        Initialize a LineCounter object.

        :param start: Point : The starting point of the line.
        :param end: Point : The ending point of the line.
        """
        self.vector = Vector(start=start, end=end)
        self.tracker_state: Dict[str, bool] = {}
        self.in_count: Dict[int, int] = {}  # Keep separate in counts for each class
        self.out_count: Dict[int, int] = {}  # Keep separate out counts for each class

    def update(self, detections: Detections):
        """
        Update the in_count and out_count for the detections that cross the line.

        :param detections: Detections : The detections for which to update the counts.
        """
        for xyxy, confidence, class_id, tracker_id in detections:
            # handle detections with no tracker_id
            if tracker_id is None:
                continue

            # initialize in_count and out_count for new class IDs
            if class_id not in self.in_count:
                self.in_count[class_id] = 0
                self.out_count[class_id] = 0

            # we check if all four anchors of bbox are on the same side of vector
            x1, y1, x2, y2 = xyxy
            rect = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            intersection = cv2.clipLine(
                rect,
                self.vector.start.as_xy_int_tuple(),
                self.vector.end.as_xy_int_tuple(),
            )

            if not intersection[0]:  # if there's no intersection, continue
                continue

            center = Point(x=(x1 + x2) / 2, y=(y1 + y2) / 2)
            is_center_above = self.vector.is_in(point=center)

            tracker_state = is_center_above

            # handle new detection
            if tracker_id not in self.tracker_state:
                self.tracker_state[tracker_id] = tracker_state
                continue

            # handle detection on the same side of the line
            if self.tracker_state.get(tracker_id) == tracker_state:
                continue

            self.tracker_state[tracker_id] = tracker_state
            # update in_count or out_count for the specific class_id
            if tracker_state:
                self.in_count[class_id] += 1
            else:
                self.out_count[class_id] += 1


class LineCounterAnnotator:
    def __init__(
        self,
        thickness: float = 2,
        color: Color = Color.white(),
        text_thickness: float = 2,
        text_color: Color = Color.black(),
        text_scale: float = 0.5,
        text_offset: float = 1.5,
        text_padding: int = 10,
    ):
        """
        Initialize the LineCounterAnnotator object with default values.

        :param thickness: float : The thickness of the line that will be drawn.
        :param color: Color : The color of the line that will be drawn.
        :param text_thickness: float : The thickness of the text that will be drawn.
        :param text_color: Color : The color of the text that will be drawn.
        :param text_scale: float : The scale of the text that will be drawn.
        :param text_offset: float : The offset of the text that will be drawn.
        :param text_padding: int : The padding of the text that will be drawn.
        """
        self.thickness: float = thickness
        self.color: Color = color
        self.text_thickness: float = text_thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_offset: float = text_offset
        self.text_padding: int = text_padding

    def annotate(self, frame: np.ndarray, line_counter: LineCounter) -> np.ndarray:
        """
        Draws the line on the frame using the line_counter provided.

        :param frame: np.ndarray : The image on which the line will be drawn
        :param line_counter: LineCounter : The line counter that will be used to draw the line
        :return: np.ndarray : The image with the line drawn on it
        """
        cv2.line(
            frame,
            line_counter.vector.start.as_xy_int_tuple(),
            line_counter.vector.end.as_xy_int_tuple(),
            self.color.as_bgr(),
            self.thickness,
            lineType=cv2.LINE_AA,
            shift=0,
        )
        cv2.circle(
            frame,
            line_counter.vector.start.as_xy_int_tuple(),
            radius=5,
            color=self.text_color.as_bgr(),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            frame,
            line_counter.vector.end.as_xy_int_tuple(),
            radius=5,
            color=self.text_color.as_bgr(),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        in_text = f"in: {line_counter.in_count}"
        out_text = f"out: {line_counter.out_count}"

        (in_text_width, in_text_height), _ = cv2.getTextSize(
            in_text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
        )
        (out_text_width, out_text_height), _ = cv2.getTextSize(
            out_text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
        )

        in_text_x = int(
            (line_counter.vector.end.x + line_counter.vector.start.x - in_text_width)
            / 2
        )
        in_text_y = int(
            (line_counter.vector.end.y + line_counter.vector.start.y + in_text_height)
            / 2
            - self.text_offset * in_text_height
        )

        out_text_x = int(
            (line_counter.vector.end.x + line_counter.vector.start.x - out_text_width)
            / 2
        )
        out_text_y = int(
            (line_counter.vector.end.y + line_counter.vector.start.y + out_text_height)
            / 2
            + self.text_offset * out_text_height
        )

        in_text_background_rect = Rect(
            x=in_text_x,
            y=in_text_y - in_text_height,
            width=in_text_width,
            height=in_text_height,
        ).pad(padding=self.text_padding)
        out_text_background_rect = Rect(
            x=out_text_x,
            y=out_text_y - out_text_height,
            width=out_text_width,
            height=out_text_height,
        ).pad(padding=self.text_padding)

        cv2.rectangle(
            frame,
            in_text_background_rect.top_left.as_xy_int_tuple(),
            in_text_background_rect.bottom_right.as_xy_int_tuple(),
            self.color.as_bgr(),
            -1,
        )
        cv2.rectangle(
            frame,
            out_text_background_rect.top_left.as_xy_int_tuple(),
            out_text_background_rect.bottom_right.as_xy_int_tuple(),
            self.color.as_bgr(),
            -1,
        )

        cv2.putText(
            frame,
            in_text,
            (in_text_x, in_text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_scale,
            self.text_color.as_bgr(),
            self.text_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            out_text,
            (out_text_x, out_text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_scale,
            self.text_color.as_bgr(),
            self.text_thickness,
            cv2.LINE_AA,
        )
