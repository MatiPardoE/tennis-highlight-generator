from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Point:
    x: int
    y: int

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Point:
        return cls(x=int(data["x"]), y=int(data["y"]))


@dataclass(slots=True)
class ROI:
    points: tuple[Point, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"points": [point.to_dict() for point in self.points]}

    @property
    def xs(self) -> list[int]:
        return [point.x for point in self.points]

    @property
    def ys(self) -> list[int]:
        return [point.y for point in self.points]

    @property
    def min_x(self) -> int:
        return min(self.xs)

    @property
    def min_y(self) -> int:
        return min(self.ys)

    @property
    def max_x(self) -> int:
        return max(self.xs)

    @property
    def max_y(self) -> int:
        return max(self.ys)

    @property
    def width(self) -> int:
        return (self.max_x - self.min_x) + 1

    @property
    def height(self) -> int:
        return (self.max_y - self.min_y) + 1

    def translated(self, dx: int, dy: int) -> ROI:
        return ROI(
            points=tuple(
                Point(point.x + dx, point.y + dy)
                for point in self.points
            )
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ROI | None:
        if data is None:
            return None
        points = tuple(Point.from_dict(item) for item in data["points"])
        if len(points) != 6:
            raise ValueError("El ROI poligonal debe tener exactamente 6 puntos.")
        return cls(points=points)


@dataclass(slots=True)
class Segment:
    start_sec: float
    end_sec: float

    def to_dict(self) -> dict[str, float]:
        return {
            "start_sec": round(float(self.start_sec), 3),
            "end_sec": round(float(self.end_sec), 3),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Segment:
        return cls(
            start_sec=float(data["start_sec"]),
            end_sec=float(data["end_sec"]),
        )


@dataclass(slots=True)
class VideoMetadata:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": round(float(self.fps), 6),
            "frame_count": int(self.frame_count),
            "duration_sec": round(float(self.duration_sec), 6),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoMetadata:
        return cls(
            path=str(data["path"]),
            width=int(data["width"]),
            height=int(data["height"]),
            fps=float(data["fps"]),
            frame_count=int(data["frame_count"]),
            duration_sec=float(data["duration_sec"]),
        )


@dataclass(slots=True)
class PreprocessConfig:
    rotation: float = 0.0
    roi: ROI | None = None
    analysis_start_sec: float = 0.0
    proxy_fps: float = 8.0
    proxy_width: int = 640
    grayscale: bool = True
    proxy_codec: str = "libx264"
    proxy_crf: int = 34
    proxy_preset: str = "ultrafast"


@dataclass(slots=True)
class DetectionConfig:
    sensitivity: float = 0.55
    smooth_window_sec: float = 1.0
    min_segment_sec: float = 4.0
    min_gap_sec: float = 2.0
    padding_sec: float = 0.35
    blur_kernel_size: int = 5
    yolo_model: str = "yolov8n.pt"
    person_confidence: float = 0.30
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


@dataclass(slots=True)
class RenderConfig:
    video_codec: str = "libx264"
    crf: int = 18
    preset: str = "medium"
    audio_codec: str = "aac"
    audio_bitrate: str = "160k"
    apply_calibrated_geometry: bool = True
    apply_roi_crop: bool = False


@dataclass(slots=True)
class PipelineManifest:
    original_video_path: str
    proxy_video_path: str
    rotation: float
    roi: ROI | None
    analysis_start_sec: float
    fps_original: float
    fps_proxy: float
    useful_segments: list[Segment] = field(default_factory=list)
    original_width: int = 0
    original_height: int = 0
    proxy_width: int = 0
    proxy_height: int = 0
    duration_original_sec: float = 0.0
    duration_proxy_sec: float = 0.0
    debug_video_path: str | None = None
    roi_mask_path: str | None = None

    @property
    def original_path(self) -> Path:
        return Path(self.original_video_path)

    @property
    def proxy_path(self) -> Path:
        return Path(self.proxy_video_path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_video_path": self.original_video_path,
            "proxy_video_path": self.proxy_video_path,
            "rotation": round(float(self.rotation), 4),
            "roi": self.roi.to_dict() if self.roi is not None else None,
            "analysis_start_sec": round(float(self.analysis_start_sec), 6),
            "fps_original": round(float(self.fps_original), 6),
            "fps_proxy": round(float(self.fps_proxy), 6),
            "useful_segments": [segment.to_dict() for segment in self.useful_segments],
            "original_width": self.original_width,
            "original_height": self.original_height,
            "proxy_width": self.proxy_width,
            "proxy_height": self.proxy_height,
            "duration_original_sec": round(float(self.duration_original_sec), 6),
            "duration_proxy_sec": round(float(self.duration_proxy_sec), 6),
            "debug_video_path": self.debug_video_path,
            "roi_mask_path": self.roi_mask_path,
        }

    def save(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineManifest:
        return cls(
            original_video_path=str(data["original_video_path"]),
            proxy_video_path=str(data["proxy_video_path"]),
            rotation=float(data["rotation"]),
            roi=ROI.from_dict(data.get("roi")),
            analysis_start_sec=float(data.get("analysis_start_sec", 0.0)),
            fps_original=float(data["fps_original"]),
            fps_proxy=float(data["fps_proxy"]),
            useful_segments=[Segment.from_dict(item) for item in data.get("useful_segments", [])],
            original_width=int(data.get("original_width", 0)),
            original_height=int(data.get("original_height", 0)),
            proxy_width=int(data.get("proxy_width", 0)),
            proxy_height=int(data.get("proxy_height", 0)),
            duration_original_sec=float(data.get("duration_original_sec", 0.0)),
            duration_proxy_sec=float(data.get("duration_proxy_sec", 0.0)),
            debug_video_path=str(data["debug_video_path"]) if data.get("debug_video_path") else None,
            roi_mask_path=str(data["roi_mask_path"]) if data.get("roi_mask_path") else None,
        )

    @classmethod
    def load(cls, input_path: Path) -> PipelineManifest:
        return cls.from_dict(json.loads(input_path.read_text(encoding="utf-8")))


def parse_roi(value: str | None) -> ROI | None:
    if value is None:
        return None
    raw_value = value.strip()
    if not raw_value:
        return None

    point_chunks = [chunk.strip() for chunk in raw_value.split(";") if chunk.strip()]
    if len(point_chunks) != 6:
        raise ValueError("ROI inválido. Usá el formato x1,y1;x2,y2;x3,y3;x4,y4;x5,y5;x6,y6.")

    points: list[Point] = []
    for chunk in point_chunks:
        parts = [part.strip() for part in chunk.split(",")]
        if len(parts) != 2:
            raise ValueError("ROI inválido. Cada punto debe tener formato x,y.")
        points.append(Point(x=int(parts[0]), y=int(parts[1])))

    return ROI(points=tuple(points))
