from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import DetectionConfig, PipelineManifest, ROI, Segment
from preprocess import roi_points_relative_to_bbox


class DetectionError(RuntimeError):
    pass


def _moving_average(values: list[float], window_size: int) -> list[float]:
    if not values:
        return []
    if window_size <= 1:
        return list(values)

    arr = np.array(values, dtype=np.float32)
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    return np.convolve(arr, kernel, mode="same").astype(float).tolist()


def _compute_activity_threshold(smoothed_scores: list[float], sensitivity: float) -> float:
    if not smoothed_scores:
        return 0.0

    arr = np.array(smoothed_scores, dtype=np.float32)
    low = float(np.percentile(arr, 20))
    high = float(np.percentile(arr, 80))
    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))

    if abs(high - low) < 1e-9:
        return high + 1e-6

    return low + (high - low) * (1.0 - sensitivity)


def _mask_to_segments(times_sec: list[float], active_mask: list[bool], sample_fps: float) -> list[Segment]:
    if not times_sec:
        return []

    sample_period = 1.0 / max(sample_fps, 0.1)
    segments: list[Segment] = []
    current_start: float | None = None

    for idx, is_active in enumerate(active_mask):
        if is_active and current_start is None:
            current_start = times_sec[idx]
        elif not is_active and current_start is not None:
            segments.append(Segment(start_sec=current_start, end_sec=times_sec[idx]))
            current_start = None

    if current_start is not None:
        segments.append(Segment(start_sec=current_start, end_sec=times_sec[-1] + sample_period))

    return segments


def _merge_close_segments(segments: list[Segment], max_gap_sec: float) -> list[Segment]:
    if not segments:
        return []

    ordered = sorted(segments, key=lambda item: item.start_sec)
    merged: list[Segment] = [Segment(ordered[0].start_sec, ordered[0].end_sec)]

    for segment in ordered[1:]:
        last = merged[-1]
        if segment.start_sec - last.end_sec <= max_gap_sec:
            last.end_sec = max(last.end_sec, segment.end_sec)
            continue
        merged.append(Segment(segment.start_sec, segment.end_sec))

    return merged


def _remove_short_segments(segments: list[Segment], min_segment_sec: float) -> list[Segment]:
    return [segment for segment in segments if segment.end_sec - segment.start_sec >= min_segment_sec]


def _apply_padding(
    segments: list[Segment],
    padding_sec: float,
    max_duration_sec: float,
) -> list[Segment]:
    padded: list[Segment] = []
    for segment in segments:
        start_sec = max(0.0, segment.start_sec - padding_sec)
        end_sec = min(max_duration_sec, segment.end_sec + padding_sec)
        if end_sec > start_sec:
            padded.append(Segment(start_sec=start_sec, end_sec=end_sec))
    return padded


def _load_person_model(model_name: str) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise DetectionError(
            "Falta la dependencia 'ultralytics'. Instalá requirements.txt para usar detección de personas."
        ) from exc

    try:
        return YOLO(model_name)
    except Exception as exc:  # noqa: BLE001
        raise DetectionError(f"No se pudo cargar el modelo YOLO '{model_name}': {exc}") from exc


def _build_roi_mask(roi: ROI | None, frame_width: int, frame_height: int) -> np.ndarray | None:
    if roi is None:
        return None
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    polygon = _roi_points_for_frame(roi, frame_width, frame_height)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def _roi_points_for_frame(roi: ROI | None, frame_width: int, frame_height: int) -> np.ndarray | None:
    if roi is None:
        return None

    polygon = roi_points_relative_to_bbox(roi).astype(np.float32)
    if roi.width <= 1 or roi.height <= 1:
        return np.clip(polygon, [0, 0], [max(0, frame_width - 1), max(0, frame_height - 1)]).astype(np.int32)

    scale_x = max(0.0, float(frame_width - 1)) / float(roi.width - 1)
    scale_y = max(0.0, float(frame_height - 1)) / float(roi.height - 1)
    polygon[:, 0] *= scale_x
    polygon[:, 1] *= scale_y
    polygon = np.rint(polygon)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, max(0, frame_width - 1))
    polygon[:, 1] = np.clip(polygon[:, 1], 0, max(0, frame_height - 1))
    return polygon.astype(np.int32)


def _clip_box_xyxy(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
    x1 = max(0, min(width - 1, int(round(float(box[0])))))
    y1 = max(0, min(height - 1, int(round(float(box[1])))))
    x2 = max(0, min(width, int(round(float(box[2])))))
    y2 = max(0, min(height, int(round(float(box[3])))))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _box_overlaps_mask(
    box: tuple[int, int, int, int],
    roi_mask: np.ndarray | None,
) -> bool:
    if roi_mask is None:
        return True
    x1, y1, x2, y2 = box
    patch = roi_mask[y1:y2, x1:x2]
    if patch.size == 0:
        return False
    return float(np.mean(patch > 0)) >= 0.15


def _detect_main_players(
    model: Any,
    frame_bgr: np.ndarray,
    cfg: DetectionConfig,
    roi_mask: np.ndarray | None,
) -> tuple[tuple[int, int, int, int], ...]:
    try:
        results = model.predict(source=frame_bgr, classes=[0], conf=cfg.person_confidence, verbose=False)
    except Exception as exc:  # noqa: BLE001
        raise DetectionError(f"Falló la detección de personas con YOLO: {exc}") from exc

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

    height, width = frame_bgr.shape[:2]
    ranked: list[tuple[int, tuple[int, int, int, int]]] = []
    for raw_box in xyxy:
        clipped = _clip_box_xyxy(raw_box, width, height)
        if clipped is None or not _box_overlaps_mask(clipped, roi_mask):
            continue
        x1, y1, x2, y2 = clipped
        area = (x2 - x1) * (y2 - y1)
        ranked.append((area, clipped))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return tuple(box for _, box in ranked[: max(1, cfg.max_players)])


def _flow_magnitude(prev_gray: np.ndarray, curr_gray: np.ndarray, cfg: DetectionConfig) -> np.ndarray:
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


def _score_from_flow_roi(flow_roi: np.ndarray, cfg: DetectionConfig) -> float:
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
    cfg: DetectionConfig,
) -> tuple[float, ...]:
    scores: list[float] = []
    for x1, y1, x2, y2 in player_boxes:
        roi = flow_mag[y1:y2, x1:x2]
        scores.append(_score_from_flow_roi(roi, cfg))
    return tuple(scores)


def _combine_player_scores(player_scores: tuple[float, ...], cfg: DetectionConfig) -> float:
    if not player_scores:
        return 0.0
    if cfg.global_activity_mode.lower().strip() == "mean":
        return float(np.mean(player_scores))
    return float(np.max(player_scores))


def extract_motion_scores(
    proxy_video_path: Path,
    cfg: DetectionConfig,
    roi: ROI | None = None,
) -> tuple[list[float], list[float], float, list[tuple[tuple[int, int, int, int], ...]], list[tuple[float, ...]]]:
    cap = cv2.VideoCapture(str(proxy_video_path))
    if not cap.isOpened():
        raise DetectionError(f"No se pudo abrir el proxy: {proxy_video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0:
        cap.release()
        raise DetectionError(f"FPS inválido en el proxy: {proxy_video_path}")

    blur_size = max(1, int(cfg.blur_kernel_size))
    if blur_size % 2 == 0:
        blur_size += 1

    person_model = _load_person_model(cfg.yolo_model)

    ok, previous_frame = cap.read()
    if not ok or previous_frame is None:
        cap.release()
        raise DetectionError("El proxy no tiene frames para analizar.")

    frame_height, frame_width = previous_frame.shape[:2]
    roi_mask = _build_roi_mask(roi, frame_width, frame_height)

    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    previous_gray = cv2.GaussianBlur(previous_gray, (blur_size, blur_size), 0)
    if roi_mask is not None:
        previous_gray = cv2.bitwise_and(previous_gray, previous_gray, mask=roi_mask)

    scores: list[float] = []
    times_sec: list[float] = []
    player_boxes_per_frame: list[tuple[tuple[int, int, int, int], ...]] = []
    player_scores_per_frame: list[tuple[float, ...]] = []
    frame_index = 1

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        if roi_mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=roi_mask)

        player_boxes = _detect_main_players(person_model, frame, cfg, roi_mask)
        flow_mag = _flow_magnitude(previous_gray, gray, cfg)
        player_scores = _player_scores_from_flow(flow_mag, player_boxes, cfg)
        global_score = _combine_player_scores(player_scores, cfg)

        times_sec.append(frame_index / fps)
        scores.append(global_score)
        player_boxes_per_frame.append(player_boxes)
        player_scores_per_frame.append(player_scores)

        previous_gray = gray
        frame_index += 1

    cap.release()
    return scores, times_sec, fps, player_boxes_per_frame, player_scores_per_frame


def export_detection_debug_video(
    proxy_video_path: Path,
    output_path: Path,
    roi: ROI | None,
    scores: list[float],
    smoothed_scores: list[float],
    threshold: float,
    player_boxes_per_frame: list[tuple[tuple[int, int, int, int], ...]],
    player_scores_per_frame: list[tuple[float, ...]],
) -> Path:
    cap = cv2.VideoCapture(str(proxy_video_path))
    if not cap.isOpened():
        raise DetectionError(f"No se pudo abrir el proxy para debug: {proxy_video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0.0 or width <= 0 or height <= 0:
        cap.release()
        raise DetectionError("Metadata inválida al exportar video debug.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise DetectionError(f"No se pudo crear el video debug: {output_path}")

    polygon = _roi_points_for_frame(roi, width, height)
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            overlay = frame.copy()
            if polygon is not None:
                cv2.polylines(overlay, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

            score_idx = max(0, min(frame_idx - 1, len(scores) - 1)) if scores else 0
            raw_score = scores[score_idx] if scores else 0.0
            smooth_score = smoothed_scores[score_idx] if smoothed_scores else 0.0
            is_active = smooth_score >= threshold
            label = "GAME" if is_active else "PAUSE"
            color = (0, 200, 0) if is_active else (0, 80, 255)

            boxes = player_boxes_per_frame[score_idx] if score_idx < len(player_boxes_per_frame) else tuple()
            box_scores = player_scores_per_frame[score_idx] if score_idx < len(player_scores_per_frame) else tuple()
            for box_idx, (x1, y1, x2, y2) in enumerate(boxes):
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                score_text = f"p{box_idx + 1}:{box_scores[box_idx]:.2f}" if box_idx < len(box_scores) else f"p{box_idx + 1}"
                cv2.putText(
                    overlay,
                    score_text,
                    (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            panel_x, panel_y = 18, 18
            panel_w = min(340, max(230, width // 4))
            panel_h = 82
            panel = overlay.copy()
            cv2.rectangle(panel, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
            overlay = cv2.addWeighted(panel, 0.55, overlay, 0.45, 0.0)

            text_x = panel_x + 12
            text_y = panel_y + 22
            font_scale = 0.46
            thickness = 1
            line_gap = 17
            cv2.putText(overlay, f"f={frame_idx}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            cv2.putText(overlay, f"players={len(boxes)} raw={raw_score:.3f}", (text_x, text_y + line_gap), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            cv2.putText(overlay, f"smooth={smooth_score:.3f}", (text_x, text_y + (2 * line_gap)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            cv2.putText(overlay, f"thr={threshold:.3f} {label}", (text_x, text_y + (3 * line_gap)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

            bar_x, bar_y, bar_w, bar_h = text_x, panel_y + panel_h - 18, panel_w - 24, 8
            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
            score_w = int(max(0.0, min(smooth_score, 1.0)) * bar_w)
            threshold_x = bar_x + int(max(0.0, min(threshold, 1.0)) * bar_w)
            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + score_w, bar_y + bar_h), color, -1)
            cv2.line(overlay, (threshold_x, bar_y - 3), (threshold_x, bar_y + bar_h + 3), (255, 255, 255), 1)

            writer.write(overlay)
            frame_idx += 1
    finally:
        writer.release()
        cap.release()

    return output_path


def detect_useful_segments(
    manifest_path: Path,
    detection_cfg: DetectionConfig,
    output_manifest_path: Path | None = None,
) -> PipelineManifest:
    manifest = PipelineManifest.load(manifest_path.expanduser().resolve())
    scores, times_sec, proxy_fps, player_boxes_per_frame, player_scores_per_frame = extract_motion_scores(
        manifest.proxy_path,
        cfg=detection_cfg,
        roi=manifest.roi,
    )
    timeline_times_sec = [manifest.analysis_start_sec + current_time for current_time in times_sec]

    if not scores:
        manifest.useful_segments = []
        save_path = output_manifest_path or manifest_path
        manifest.save(save_path.expanduser().resolve())
        return manifest

    smooth_window = max(1, int(round(detection_cfg.smooth_window_sec * proxy_fps)))
    smoothed_scores = _moving_average(scores, smooth_window)
    threshold = _compute_activity_threshold(smoothed_scores, detection_cfg.sensitivity)
    active_mask = [score >= threshold for score in smoothed_scores]

    raw_segments = _mask_to_segments(timeline_times_sec, active_mask, proxy_fps)
    merged_segments = _merge_close_segments(raw_segments, detection_cfg.min_gap_sec)
    filtered_segments = _remove_short_segments(merged_segments, detection_cfg.min_segment_sec)
    padded_segments = _apply_padding(
        filtered_segments,
        padding_sec=detection_cfg.padding_sec,
        max_duration_sec=manifest.duration_original_sec,
    )
    manifest.useful_segments = _merge_close_segments(padded_segments, max_gap_sec=0.01)

    debug_video_path = manifest.proxy_path.parent / "debug_detection.mp4"
    export_detection_debug_video(
        proxy_video_path=manifest.proxy_path,
        output_path=debug_video_path,
        roi=manifest.roi,
        scores=scores,
        smoothed_scores=smoothed_scores,
        threshold=threshold,
        player_boxes_per_frame=player_boxes_per_frame,
        player_scores_per_frame=player_scores_per_frame,
    )
    manifest.debug_video_path = str(debug_video_path.resolve())

    save_path = output_manifest_path or manifest_path
    manifest.save(save_path.expanduser().resolve())
    return manifest
