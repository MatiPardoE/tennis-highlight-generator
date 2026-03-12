from __future__ import annotations

import numpy as np

from src.config import ProcessingConfig
from src.models import MotionSeries, TimeSegment


def _compute_activity_threshold(smoothed_scores: list[float], sensitivity: float) -> float:
    arr = np.array(smoothed_scores, dtype=np.float32)
    low = float(np.percentile(arr, 20))
    high = float(np.percentile(arr, 80))
    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))

    # Mayor sensibilidad => umbral más bajo => más segmentos detectados como GAME.
    return low + (high - low) * (1.0 - sensitivity)


def _mask_to_segments(times_sec: list[float], active_mask: list[bool], sample_fps: float) -> list[TimeSegment]:
    if not times_sec:
        return []

    sample_period = 1.0 / max(sample_fps, 0.1)
    segments: list[TimeSegment] = []

    start: float | None = None
    for idx, is_active in enumerate(active_mask):
        if is_active and start is None:
            start = times_sec[idx]
        if not is_active and start is not None:
            end = times_sec[idx]
            segments.append(TimeSegment(start_sec=start, end_sec=end))
            start = None

    if start is not None:
        segments.append(TimeSegment(start_sec=start, end_sec=times_sec[-1] + sample_period))

    return segments


def detect_initial_segments(motion: MotionSeries, cfg: ProcessingConfig) -> tuple[float, list[TimeSegment]]:
    threshold = _compute_activity_threshold(motion.smoothed_scores, cfg.sensitivity)
    active_mask = [score >= threshold for score in motion.smoothed_scores]
    segments = _mask_to_segments(motion.times_sec, active_mask, motion.sample_fps)
    return threshold, segments
