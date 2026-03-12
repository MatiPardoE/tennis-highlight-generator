from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class VideoMetadata:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float


@dataclass(slots=True)
class MotionSeries:
    times_sec: list[float]
    scores: list[float]
    smoothed_scores: list[float]
    sample_fps: float
    player_boxes: list[tuple[tuple[int, int, int, int], ...]] = field(default_factory=list)
    player_scores: list[tuple[float, ...]] = field(default_factory=list)
    profiling: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class TimeSegment:
    start_sec: float
    end_sec: float
    label: str = "GAME"

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass(slots=True)
class DetectionResult:
    threshold: float
    initial_segments: list[TimeSegment]
    final_segments: list[TimeSegment]
