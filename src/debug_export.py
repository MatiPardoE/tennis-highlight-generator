from __future__ import annotations

from pathlib import Path

import cv2

from src.config import ProcessingConfig
from src.models import MotionSeries, VideoMetadata
from src.motion_features import resize_preserving_aspect, sample_step_from_fps


class DebugExportError(RuntimeError):
    pass


def export_player_activity_debug_video(
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
    frame_index = 0
    sampled_frame_index = 0
    writer: cv2.VideoWriter | None = None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % sample_step_frames != 0:
            frame_index += 1
            continue

        resized = resize_preserving_aspect(frame, cfg.resize_width)
        if sampled_frame_index == 0:
            sampled_frame_index += 1
            frame_index += 1
            continue

        score_index = sampled_frame_index - 1
        if score_index >= len(motion.scores):
            break

        debug_frame = resized.copy()
        player_boxes = motion.player_boxes[score_index] if score_index < len(motion.player_boxes) else tuple()
        player_scores = motion.player_scores[score_index] if score_index < len(motion.player_scores) else tuple()

        for player_idx, (x1, y1, x2, y2) in enumerate(player_boxes):
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            score = player_scores[player_idx] if player_idx < len(player_scores) else 0.0
            label = f"P{player_idx + 1}: {score:.3f}"
            cv2.putText(
                debug_frame,
                label,
                (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        raw_score = motion.scores[score_index]
        smooth_score = motion.smoothed_scores[score_index]
        is_active = smooth_score >= activity_threshold
        status = "GAME" if is_active else "PAUSA"
        text_color = (0, 255, 0) if is_active else (0, 165, 255)

        cv2.putText(
            debug_frame,
            f"Score: {raw_score:.3f}  Smooth: {smooth_score:.3f}  Thr: {activity_threshold:.3f}",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug_frame,
            f"Estado: {status}",
            (12, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            text_color,
            2,
            cv2.LINE_AA,
        )

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
        sampled_frame_index += 1
        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()

    if writer is None:
        raise DebugExportError("No se pudieron generar frames para el video debug.")

    return output_path
