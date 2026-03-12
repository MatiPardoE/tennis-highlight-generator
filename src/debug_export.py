from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.config import ProcessingConfig
from src.motion_features import preprocess_frame_for_motion, sample_step_from_fps
from src.models import MotionSeries, VideoMetadata


class DebugExportError(RuntimeError):
    pass


def pixel_diff_threshold_from_activity(activity_threshold: float) -> int:
    clipped = float(np.clip(activity_threshold, 0.0, 1.0))
    return int(np.clip(round(clipped * 255.0), 1, 255))


def export_threshold_motion_video(
    input_path: Path,
    output_path: Path,
    metadata: VideoMetadata,
    motion: MotionSeries,
    cfg: ProcessingConfig,
    activity_threshold: float,
) -> Path:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise DebugExportError(f"No se pudo abrir el video para debug: {input_path}")

    sample_step_frames = sample_step_from_fps(metadata.fps, cfg.sample_fps)
    output_fps = metadata.fps / sample_step_frames
    pixel_threshold = pixel_diff_threshold_from_activity(activity_threshold)

    frame_index = 0
    sample_index = 0
    previous_gray: np.ndarray | None = None
    writer: cv2.VideoWriter | None = None

    output_path.parent.mkdir(parents=True, exist_ok=True)

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
            _, motion_mask = cv2.threshold(diff, pixel_threshold, 255, cv2.THRESH_BINARY)

            score_active = sample_index < len(motion.smoothed_scores) and motion.smoothed_scores[sample_index] >= activity_threshold
            if not score_active:
                motion_mask = np.zeros_like(motion_mask)

            debug_frame = np.zeros((motion_mask.shape[0], motion_mask.shape[1], 3), dtype=np.uint8)
            debug_frame[:, :, 1] = motion_mask

            if writer is None:
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    output_fps,
                    (debug_frame.shape[1], debug_frame.shape[0]),
                )
                if not writer.isOpened():
                    cap.release()
                    raise DebugExportError(f"No se pudo crear el video debug: {output_path}")

            writer.write(debug_frame)
            sample_index += 1

        previous_gray = gray
        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()

    if writer is None:
        raise DebugExportError("No se pudieron generar frames para el video debug.")

    return output_path
