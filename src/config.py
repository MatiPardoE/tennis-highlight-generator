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
    debug_mode: bool = False


@dataclass(slots=True)
class ExportConfig:
    crf: int = 20
    preset: str = "veryfast"
