from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ProcessingConfig:
    sensitivity: float = 0.55
    min_game_sec: float = 4.0
    min_pause_sec: float = 2.0
    padding_sec: float = 0.35
    sample_fps: float = 5.0
    smooth_window_sec: float = 1.0
    resize_width: int = 640
    yolo_model: str = "yolov8n.pt"
    person_confidence: float = 0.35
    max_players: int = 2
    flow_metric: str = "p90"
    flow_fast_threshold: float = 2.0
    global_activity_mode: str = "max"
    farneback_pyr_scale: float = 0.5
    farneback_levels: int = 3
    farneback_winsize: int = 15
    farneback_iterations: int = 3
    farneback_poly_n: int = 5
    farneback_poly_sigma: float = 1.2
    debug_mode: bool = False


@dataclass(slots=True)
class ExportConfig:
    crf: int = 20
    preset: str = "veryfast"
