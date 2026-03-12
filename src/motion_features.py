from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.config import ProcessingConfig
from src.models import MotionSeries, VideoMetadata


class MotionExtractionError(RuntimeError):
    pass


def resize_preserving_aspect(frame: np.ndarray, target_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    ratio = target_width / float(w)
    target_height = int(h * ratio)
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def preprocess_frame_for_motion(frame: np.ndarray, resize_width: int) -> np.ndarray:
    resized = resize_preserving_aspect(frame, resize_width)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def _moving_average(values: list[float], window_size: int) -> list[float]:
    if not values:
        return []
    if window_size <= 1:
        return list(values)

    arr = np.array(values, dtype=np.float32)
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    smoothed = np.convolve(arr, kernel, mode="same")
    return smoothed.astype(float).tolist()


def extract_motion_series(video_path: Path, metadata: VideoMetadata, cfg: ProcessingConfig) -> MotionSeries:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise MotionExtractionError(f"No se pudo abrir el video: {video_path}")

    sample_step_frames = sample_step_from_fps(metadata.fps, cfg.sample_fps)
    effective_sample_fps = metadata.fps / sample_step_frames

    frame_index = 0
    previous_gray: np.ndarray | None = None
    times_sec: list[float] = []
    scores: list[float] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % sample_step_frames != 0:
            frame_index += 1
            continue

        gray = preprocess_frame_for_motion(frame, cfg.resize_width)
        if previous_gray is not None:
            diff = cv2.absdiff(gray, previous_gray)
            score = float(np.mean(diff) / 255.0)
            times_sec.append(frame_index / metadata.fps)
            scores.append(score)

        previous_gray = gray
        frame_index += 1

    cap.release()

    if not scores:
        raise MotionExtractionError("No se pudieron extraer features de movimiento.")

    smooth_window = max(1, int(round(cfg.smooth_window_sec * effective_sample_fps)))
    smoothed_scores = _moving_average(scores, smooth_window)

    return MotionSeries(
        times_sec=times_sec,
        scores=scores,
        smoothed_scores=smoothed_scores,
        sample_fps=effective_sample_fps,
    )


def sample_step_from_fps(video_fps: float, sample_fps: float) -> int:
    return max(1, round(video_fps / max(sample_fps, 0.1)))
