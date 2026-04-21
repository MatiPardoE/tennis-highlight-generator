from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

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


def _load_person_model(model_name: str) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise MotionExtractionError(
            "Falta la dependencia 'ultralytics'. Instalá requirements.txt para usar detección de personas."
        ) from exc

    try:
        return YOLO(model_name)
    except Exception as exc:  # noqa: BLE001
        raise MotionExtractionError(f"No se pudo cargar el modelo YOLO '{model_name}': {exc}") from exc


def _clip_box_xyxy(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
    x1 = max(0, min(width - 1, int(round(float(box[0])))))
    y1 = max(0, min(height - 1, int(round(float(box[1])))))
    x2 = max(0, min(width, int(round(float(box[2])))))
    y2 = max(0, min(height, int(round(float(box[3])))))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _detect_main_players(model: Any, frame: np.ndarray, cfg: ProcessingConfig) -> tuple[tuple[int, int, int, int], ...]:
    try:
        results = model.predict(source=frame, classes=[0], conf=cfg.person_confidence, verbose=False)
    except Exception as exc:  # noqa: BLE001
        raise MotionExtractionError(f"Falló la detección de personas con YOLO: {exc}") from exc

    if not results:
        return tuple()

    result = results[0]
    if result.boxes is None or result.boxes.xyxy is None:
        return tuple()

    xyxy = result.boxes.xyxy
    if hasattr(xyxy, "cpu"):
        xyxy = xyxy.cpu().numpy()
    else:
        xyxy = np.asarray(xyxy)

    height, width = frame.shape[:2]
    ranked: list[tuple[int, tuple[int, int, int, int]]] = []

    for raw_box in xyxy:
        clipped = _clip_box_xyxy(raw_box, width, height)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        area = (x2 - x1) * (y2 - y1)
        ranked.append((area, clipped))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return tuple(box for _, box in ranked[: max(1, cfg.max_players)])


def _flow_magnitude(prev_gray: np.ndarray, curr_gray: np.ndarray, cfg: ProcessingConfig) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        cfg.farneback_pyr_scale,
        cfg.farneback_levels,
        cfg.farneback_winsize,
        cfg.farneback_iterations,
        cfg.farneback_poly_n,
        cfg.farneback_poly_sigma,
        0,
    )
    return cv2.magnitude(flow[..., 0], flow[..., 1])


def _score_from_flow_roi(flow_roi: np.ndarray, cfg: ProcessingConfig) -> float:
    if flow_roi.size == 0:
        return 0.0

    metric = cfg.flow_metric.lower().strip()
    if metric == "mean":
        return float(np.mean(flow_roi))
    if metric in {"fast_ratio", "count_fast"}:
        return float(np.mean(flow_roi >= cfg.flow_fast_threshold))
    return float(np.percentile(flow_roi, 90))


def _player_scores_from_flow(
    flow_mag: np.ndarray,
    player_boxes: tuple[tuple[int, int, int, int], ...],
    cfg: ProcessingConfig,
) -> tuple[float, ...]:
    scores: list[float] = []
    for x1, y1, x2, y2 in player_boxes:
        roi = flow_mag[y1:y2, x1:x2]
        scores.append(_score_from_flow_roi(roi, cfg))
    return tuple(scores)


def _combine_player_scores(player_scores: tuple[float, ...], cfg: ProcessingConfig) -> float:
    if not player_scores:
        return 0.0
    if cfg.global_activity_mode.lower().strip() == "mean":
        return float(np.mean(player_scores))
    return float(np.max(player_scores))


def extract_motion_series(
    video_path: Path,
    metadata: VideoMetadata,
    cfg: ProcessingConfig,
    logger: logging.Logger | None = None,
) -> MotionSeries:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise MotionExtractionError(f"No se pudo abrir el video: {video_path}")

    extraction_started_at = perf_counter()
    person_model = _load_person_model(cfg.yolo_model)
    sample_step_frames = sample_step_from_fps(metadata.fps, cfg.sample_fps)
    effective_sample_fps = metadata.fps / sample_step_frames
    sampled_frames_estimate = max(1, metadata.frame_count // sample_step_frames)
    sampled_pairs_estimate = max(1, sampled_frames_estimate - 1)
    progress_interval_pairs = max(1, sampled_pairs_estimate // 10)

    frame_index = 0
    previous_gray: np.ndarray | None = None
    times_sec: list[float] = []
    scores: list[float] = []
    player_boxes_per_frame: list[tuple[tuple[int, int, int, int], ...]] = []
    player_scores_per_frame: list[tuple[float, ...]] = []
    processed_pairs = 0
    players_detected_total = 0
    detection_seconds = 0.0
    flow_seconds = 0.0
    score_seconds = 0.0

    if logger is not None:
        logger.info(
            "Inicia extracción de movimiento. fps=%.2f frame_count=%d sample_step=%d (~%d pares).",
            metadata.fps,
            metadata.frame_count,
            sample_step_frames,
            sampled_pairs_estimate,
        )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % sample_step_frames != 0:
                frame_index += 1
                continue

            resized = resize_preserving_aspect(frame, cfg.resize_width)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if previous_gray is not None:
                started_at = perf_counter()
                player_boxes = _detect_main_players(person_model, resized, cfg)
                detection_seconds += perf_counter() - started_at

                started_at = perf_counter()
                flow_mag = _flow_magnitude(previous_gray, gray, cfg)
                flow_seconds += perf_counter() - started_at

                started_at = perf_counter()
                player_scores = _player_scores_from_flow(flow_mag, player_boxes, cfg)
                global_score = _combine_player_scores(player_scores, cfg)
                score_seconds += perf_counter() - started_at

                times_sec.append(frame_index / metadata.fps)
                scores.append(global_score)
                player_boxes_per_frame.append(player_boxes)
                player_scores_per_frame.append(player_scores)
                processed_pairs += 1
                players_detected_total += len(player_boxes)

                if logger is not None and processed_pairs % progress_interval_pairs == 0:
                    progress = (processed_pairs / sampled_pairs_estimate) * 100.0
                    logger.info(
                        "Extracción progreso: %d/%d pares (%.1f%%).",
                        processed_pairs,
                        sampled_pairs_estimate,
                        min(progress, 100.0),
                    )

            previous_gray = gray
            frame_index += 1
    finally:
        cap.release()

    if not scores:
        raise MotionExtractionError("No se pudieron extraer features de movimiento.")

    smooth_window = max(1, int(round(cfg.smooth_window_sec * effective_sample_fps)))
    smoothed_scores = _moving_average(scores, smooth_window)
    extraction_seconds = perf_counter() - extraction_started_at
    mean_players = players_detected_total / max(1, processed_pairs)

    profiling = {
        "extract_seconds": extraction_seconds,
        "sample_step_frames": float(sample_step_frames),
        "sampled_pairs": float(processed_pairs),
        "throughput_pairs_per_sec": processed_pairs / max(extraction_seconds, 1e-6),
        "detection_seconds": detection_seconds,
        "flow_seconds": flow_seconds,
        "scoring_seconds": score_seconds,
        "detection_share": detection_seconds / max(extraction_seconds, 1e-6),
        "flow_share": flow_seconds / max(extraction_seconds, 1e-6),
        "mean_players_detected": mean_players,
        "effective_sample_fps": effective_sample_fps,
    }

    if logger is not None:
        logger.info(
            "Extracción finalizada en %.2fs. pares=%d throughput=%.2f pares/s yolo=%.2fs flow=%.2fs.",
            extraction_seconds,
            processed_pairs,
            profiling["throughput_pairs_per_sec"],
            detection_seconds,
            flow_seconds,
        )

    return MotionSeries(
        times_sec=times_sec,
        scores=scores,
        smoothed_scores=smoothed_scores,
        sample_fps=effective_sample_fps,
        player_boxes=player_boxes_per_frame,
        player_scores=player_scores_per_frame,
        profiling=profiling,
    )


def sample_step_from_fps(video_fps: float, sample_fps: float) -> int:
    return max(1, round(video_fps / max(sample_fps, 0.1)))
