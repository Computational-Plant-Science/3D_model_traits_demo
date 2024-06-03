from pathlib import Path


class DIRT3DOptions:
    def __init__(
            self,
            input_file: str,
            output_directory: str,
            interval: int,
            direction: str,
            reverse: bool,
            frames: int,
            threshold: float,
            distance_threshold: int,
            distance_tracking: float,
            distance_ratio: float,
            max_skipped_frames: int,
            max_trace_length: int,
            min_radius: int,
            max_radius: int,
            min_angle: int):
        self.input_file = input_file
        self.input_name = Path(input_file).name
        self.input_name = Path(input_file).stem
        self.output_directory = output_directory
        self.interval = interval
        self.direction = direction
        self.reverse = reverse
        self.frames = frames
        self.threshold = threshold
        self.distance_threshold = distance_threshold
        self.distance_tracking = distance_tracking
        self.distance_ratio = distance_ratio
        self.max_skipped_frames = max_skipped_frames
        self.max_trace_length = max_trace_length
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_angle = min_angle
