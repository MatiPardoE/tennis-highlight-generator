from __future__ import annotations

import math
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from config import PipelineManifest, Point, PreprocessConfig, ROI, VideoMetadata


class PreprocessError(RuntimeError):
    pass


RIGHT_ANGLE_EPSILON = 1e-3
ProgressCallback = Callable[[str, float | None], None]


def ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise PreprocessError("No se encontró ffmpeg en PATH.")


def read_video_metadata(video_path: Path) -> VideoMetadata:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise PreprocessError(f"No se pudo abrir el video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0.0:
        raise PreprocessError(f"FPS inválido en el video: {video_path}")

    duration_sec = frame_count / fps if frame_count > 0 else 0.0
    return VideoMetadata(
        path=str(video_path.resolve()),
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration_sec,
    )


def normalize_rotation(rotation: float) -> float:
    normalized = float(rotation) % 360.0
    if normalized > 180.0:
        normalized -= 360.0
    if abs(normalized) < RIGHT_ANGLE_EPSILON:
        return 0.0
    return normalized


def _is_close_angle(angle_deg: float, target_deg: float) -> bool:
    return abs(normalize_rotation(angle_deg) - target_deg) < RIGHT_ANGLE_EPSILON


def _ensure_even(value: int) -> int:
    if value <= 2:
        return 2
    return value if value % 2 == 0 else value - 1


def _rotated_rect_with_max_area(width: int, height: int, angle_rad: float) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise PreprocessError("Dimensiones inválidas para calcular el recorte de rotación.")

    sin_a = abs(math.sin(angle_rad))
    cos_a = abs(math.cos(angle_rad))

    if sin_a < 1e-10:
        return width, height
    if cos_a < 1e-10:
        return height, width

    width_is_longer = width >= height
    side_long, side_short = (width, height) if width_is_longer else (height, width)

    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        half_short = 0.5 * side_short
        if width_is_longer:
            crop_width = half_short / sin_a
            crop_height = half_short / cos_a
        else:
            crop_width = half_short / cos_a
            crop_height = half_short / sin_a
    else:
        cos_2a = (cos_a * cos_a) - (sin_a * sin_a)
        crop_width = (width * cos_a - height * sin_a) / cos_2a
        crop_height = (height * cos_a - width * sin_a) / cos_2a

    return _ensure_even(int(math.floor(crop_width))), _ensure_even(int(math.floor(crop_height)))


def _get_right_angle_dimensions(width: int, height: int, rotation_deg: float) -> tuple[int, int]:
    if _is_close_angle(rotation_deg, 90.0) or _is_close_angle(rotation_deg, -90.0):
        return height, width
    return width, height


def get_effective_frame_dimensions(width: int, height: int, rotation_deg: float) -> tuple[int, int]:
    angle = normalize_rotation(rotation_deg)
    if _is_close_angle(angle, 0.0) or _is_close_angle(angle, 180.0) or _is_close_angle(angle, -180.0):
        return width, height
    if _is_close_angle(angle, 90.0) or _is_close_angle(angle, -90.0):
        return height, width
    return _rotated_rect_with_max_area(width, height, math.radians(abs(angle)))


def validate_roi(roi: ROI | None, frame_width: int, frame_height: int) -> ROI | None:
    if roi is None:
        return None
    if roi.width <= 0 or roi.height <= 0:
        raise PreprocessError("El ROI poligonal debe abarcar un área válida.")
    for point in roi.points:
        if point.x < 0 or point.y < 0:
            raise PreprocessError("Los puntos del ROI no pueden tener coordenadas negativas.")
        if point.x >= frame_width or point.y >= frame_height:
            raise PreprocessError("Los puntos del ROI quedan fuera del frame calibrado.")
    if roi.min_x < 0 or roi.min_y < 0 or roi.max_x >= frame_width or roi.max_y >= frame_height:
        raise PreprocessError(
            "El ROI queda fuera del frame calibrado. Ajustá el polígono o el ángulo."
        )
    return roi


def roi_points_relative_to_bbox(roi: ROI) -> np.ndarray:
    return np.array(
        [[point.x - roi.min_x, point.y - roi.min_y] for point in roi.points],
        dtype=np.int32,
    )


def create_roi_mask_image(roi: ROI, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((roi.height, roi.width, 3), dtype=np.uint8)
    polygon = roi_points_relative_to_bbox(roi)
    cv2.fillPoly(mask, [polygon], (255, 255, 255))
    if not cv2.imwrite(str(output_path), mask):
        raise PreprocessError(f"No se pudo guardar la máscara ROI: {output_path}")
    return output_path


def _build_rotation_filters(
    source_width: int,
    source_height: int,
    rotation_deg: float,
) -> tuple[list[str], int, int]:
    angle = normalize_rotation(rotation_deg)

    if _is_close_angle(angle, 0.0):
        return [], source_width, source_height
    if _is_close_angle(angle, 90.0):
        return ["transpose=1"], source_height, source_width
    if _is_close_angle(angle, -90.0):
        return ["transpose=2"], source_height, source_width
    if _is_close_angle(abs(angle), 180.0):
        return ["hflip,vflip"], source_width, source_height

    crop_width, crop_height = _rotated_rect_with_max_area(
        source_width,
        source_height,
        math.radians(abs(angle)),
    )
    # OpenCV preview uses positive angles counter-clockwise. The ffmpeg rotate
    # filter uses the opposite sign convention for this workflow, so we invert
    # here to keep preview, proxy, debug and final render aligned.
    angle_rad = math.radians(-angle)
    rotate_filter = (
        f"rotate={angle_rad:.10f}:ow=rotw({angle_rad:.10f}):"
        f"oh=roth({angle_rad:.10f}):fillcolor=black@0"
    )
    crop_filter = f"crop={crop_width}:{crop_height}:(iw-{crop_width})/2:(ih-{crop_height})/2"
    return [rotate_filter, crop_filter], crop_width, crop_height


def build_geometry_filter_chain(
    source_width: int,
    source_height: int,
    rotation: float,
    roi: ROI | None,
) -> tuple[str, int, int]:
    filters, rotated_width, rotated_height = _build_rotation_filters(
        source_width,
        source_height,
        rotation,
    )

    working_width = rotated_width
    working_height = rotated_height
    if roi is not None:
        filters.append(f"crop={roi.width}:{roi.height}:{roi.min_x}:{roi.min_y}")
        working_width = roi.width
        working_height = roi.height

    return ",".join(filters), working_width, working_height


def _build_filter_chain(
    source_width: int,
    source_height: int,
    rotation: float,
    roi: ROI | None,
    proxy_width: int,
    proxy_fps: float,
    grayscale: bool,
) -> tuple[str, int, int]:
    geometry_filters, working_width, working_height = build_geometry_filter_chain(
        source_width=source_width,
        source_height=source_height,
        rotation=rotation,
        roi=roi,
    )
    filters: list[str] = [geometry_filters] if geometry_filters else []

    if proxy_width > 0 and working_width > proxy_width:
        filters.append(f"scale={proxy_width}:-2:flags=lanczos")

    if grayscale:
        filters.append("format=gray")

    filters.append(f"fps={proxy_fps}")
    filters.append("setsar=1")
    return ",".join(filters), working_width, working_height


def _build_proxy_filter_complex(
    source_width: int,
    source_height: int,
    rotation: float,
    roi: ROI | None,
    analysis_start_sec: float,
    analysis_duration_sec: float,
    proxy_width: int,
    proxy_fps: float,
    grayscale: bool,
    use_mask: bool,
) -> str:
    geometry_filters, _, _ = build_geometry_filter_chain(
        source_width=source_width,
        source_height=source_height,
        rotation=rotation,
        roi=roi,
    )
    parts: list[str] = []
    source_filters: list[str] = []
    if analysis_start_sec > 0.0:
        source_filters.append(f"trim=start={analysis_start_sec:.6f}")
        source_filters.append("setpts=PTS-STARTPTS")
    if geometry_filters:
        source_filters.append(geometry_filters)

    if source_filters:
        parts.append(f"[0:v]{','.join(source_filters)}[geom]")
    else:
        parts.append("[0:v]null[geom]")

    current_label = "geom"
    if use_mask:
        parts.append(
            "[1:v]"
            f"trim=duration={analysis_duration_sec:.6f},"
            "setpts=PTS-STARTPTS,"
            "format=rgb24[mask]"
        )
        parts.append(f"[{current_label}]format=rgb24[geomrgb]")
        parts.append("[geomrgb][mask]blend=all_mode=multiply[masked]")
        current_label = "masked"

    tail_filters: list[str] = []
    if proxy_width > 0:
        tail_filters.append(f"scale='min(iw,{proxy_width})':-2:flags=fast_bilinear")
    if grayscale:
        tail_filters.append("format=gray")
    tail_filters.append(f"fps={proxy_fps}")
    tail_filters.append("setsar=1")
    parts.append(f"[{current_label}]{','.join(tail_filters)}[vout]")
    return ";".join(parts)


def _parse_ffmpeg_time(raw_value: str) -> float | None:
    try:
        hours_str, minutes_str, seconds_str = raw_value.strip().split(":")
        return (
            int(hours_str) * 3600.0
            + int(minutes_str) * 60.0
            + float(seconds_str)
        )
    except (ValueError, TypeError):
        return None


def _run_ffmpeg(
    command: list[str],
    duration_sec: float | None = None,
    progress_callback: ProgressCallback | None = None,
) -> None:
    proc = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    stderr_lines: list[str] = []
    last_progress_bucket = -1

    if progress_callback is not None:
        progress_callback("Lanzando ffmpeg para generar el proxy...", 0.0)

    assert proc.stderr is not None
    for raw_line in proc.stderr:
        line = raw_line.strip()
        if line:
            stderr_lines.append(line)

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        if key == "out_time":
            out_time_sec = _parse_ffmpeg_time(value)
            if out_time_sec is None or duration_sec is None or duration_sec <= 0.0:
                continue

            progress = min(max(out_time_sec / duration_sec, 0.0), 1.0)
            progress_bucket = int(progress * 100)
            if progress_bucket == last_progress_bucket:
                continue

            last_progress_bucket = progress_bucket
            if progress_callback is not None:
                progress_callback(
                    f"Generando proxy... {progress_bucket}% ({out_time_sec:.1f}s / {duration_sec:.1f}s)",
                    progress,
                )
        elif key == "progress" and value == "end" and progress_callback is not None:
            progress_callback("Proxy generado. Validando archivo de salida...", 1.0)

    return_code = proc.wait()
    if return_code == 0:
        return

    stderr_tail = "\n".join(stderr_lines[-20:])
    raise PreprocessError(f"ffmpeg falló durante el preprocesado:\n{stderr_tail}")


def generate_proxy(
    original_video_path: Path,
    workspace_dir: Path,
    preprocess_cfg: PreprocessConfig,
    manifest_path: Path | None = None,
    proxy_filename: str = "proxy.mp4",
    progress_callback: ProgressCallback | None = None,
) -> PipelineManifest:
    original_path = original_video_path.expanduser().resolve()
    if not original_path.exists():
        raise PreprocessError(f"No existe el video original: {original_path}")

    ensure_ffmpeg_available()
    workspace_dir.mkdir(parents=True, exist_ok=True)

    original_metadata = read_video_metadata(original_path)
    if progress_callback is not None:
        progress_callback(
            f"Video original: {original_metadata.width}x{original_metadata.height}, "
            f"{original_metadata.fps:.2f} fps, {original_metadata.duration_sec:.1f}s",
            0.0,
        )

    analysis_start_sec = max(0.0, float(preprocess_cfg.analysis_start_sec))
    if analysis_start_sec >= original_metadata.duration_sec:
        raise PreprocessError(
            "El frame de calibración queda fuera de la duración del video. "
            "Elegí un instante anterior al final."
        )
    analysis_duration_sec = max(0.0, original_metadata.duration_sec - analysis_start_sec)

    rotation = normalize_rotation(preprocess_cfg.rotation)
    frame_width, frame_height = get_effective_frame_dimensions(
        original_metadata.width,
        original_metadata.height,
        rotation,
    )
    roi = validate_roi(preprocess_cfg.roi, frame_width, frame_height)
    roi_mask_path: Path | None = None

    proxy_path = (workspace_dir / proxy_filename).resolve()
    if roi is not None:
        roi_mask_path = create_roi_mask_image(roi, (workspace_dir / "roi_mask.png").resolve())
    filter_complex = _build_proxy_filter_complex(
        source_width=original_metadata.width,
        source_height=original_metadata.height,
        rotation=rotation,
        roi=roi,
        analysis_start_sec=analysis_start_sec,
        analysis_duration_sec=analysis_duration_sec,
        proxy_width=preprocess_cfg.proxy_width,
        proxy_fps=preprocess_cfg.proxy_fps,
        grayscale=preprocess_cfg.grayscale,
        use_mask=roi_mask_path is not None,
    )
    if progress_callback is not None:
        progress_callback(
            f"Filtro proxy listo. Inicio útil: {analysis_start_sec:.2f}s. "
            f"Área útil: {frame_width}x{frame_height}. "
            f"Proxy objetivo: {preprocess_cfg.proxy_width}px @ {preprocess_cfg.proxy_fps:.1f} fps",
            0.0,
        )

    command = [
        "ffmpeg",
        "-y",
        "-threads",
        "0",
        "-hwaccel",
        "auto",
        "-i",
        str(original_path),
    ]
    if roi_mask_path is not None:
        command.extend(["-loop", "1", "-i", str(roi_mask_path)])
    command.extend([
        "-progress",
        "pipe:2",
        "-nostats",
        "-an",
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-c:v",
        preprocess_cfg.proxy_codec,
        "-preset",
        preprocess_cfg.proxy_preset,
        "-tune",
        "fastdecode",
        "-crf",
        str(preprocess_cfg.proxy_crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(proxy_path),
    ])
    _run_ffmpeg(
        command,
        duration_sec=analysis_duration_sec,
        progress_callback=progress_callback,
    )

    proxy_metadata = read_video_metadata(proxy_path)
    manifest = PipelineManifest(
        original_video_path=str(original_path),
        proxy_video_path=str(proxy_path),
        rotation=rotation,
        roi=roi,
        analysis_start_sec=analysis_start_sec,
        fps_original=original_metadata.fps,
        fps_proxy=proxy_metadata.fps,
        useful_segments=[],
        original_width=original_metadata.width,
        original_height=original_metadata.height,
        proxy_width=proxy_metadata.width,
        proxy_height=proxy_metadata.height,
        duration_original_sec=original_metadata.duration_sec,
        duration_proxy_sec=proxy_metadata.duration_sec,
        roi_mask_path=str(roi_mask_path) if roi_mask_path is not None else None,
    )

    if manifest_path is None:
        manifest_path = workspace_dir / "segments_manifest.json"
    manifest.save(manifest_path.resolve())
    return manifest


def _rotate_frame(frame: np.ndarray, rotation_deg: float) -> np.ndarray:
    angle = normalize_rotation(rotation_deg)
    if _is_close_angle(angle, 0.0):
        return frame.copy()
    if _is_close_angle(angle, 90.0):
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if _is_close_angle(angle, -90.0):
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if _is_close_angle(abs(angle), 180.0):
        return cv2.rotate(frame, cv2.ROTATE_180)

    height, width = frame.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_a = abs(matrix[0, 0])
    sin_a = abs(matrix[0, 1])

    bound_width = int(math.ceil((height * sin_a) + (width * cos_a)))
    bound_height = int(math.ceil((height * cos_a) + (width * sin_a)))

    matrix[0, 2] += (bound_width / 2.0) - center[0]
    matrix[1, 2] += (bound_height / 2.0) - center[1]

    rotated = cv2.warpAffine(
        frame,
        matrix,
        (bound_width, bound_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    crop_width, crop_height = _rotated_rect_with_max_area(width, height, math.radians(abs(angle)))
    crop_x = max(0, (bound_width - crop_width) // 2)
    crop_y = max(0, (bound_height - crop_height) // 2)
    return rotated[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]


def _draw_roi(frame: np.ndarray, roi: ROI | None) -> np.ndarray:
    if roi is None:
        return frame
    preview = frame.copy()
    polygon = np.array([[point.x, point.y] for point in roi.points], dtype=np.int32)
    cv2.polylines(preview, [polygon], isClosed=True, color=(0, 255, 0), thickness=3)
    for index, point in enumerate(roi.points, start=1):
        cv2.circle(preview, (point.x, point.y), 6, (255, 128, 0), -1)
        cv2.putText(
            preview,
            str(index),
            (point.x + 8, point.y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 128, 0),
            2,
            cv2.LINE_AA,
        )
    return preview


def read_preview_frame(
    video_path: Path,
    frame_index: int = 0,
    rotation: float = 0.0,
    roi: ROI | None = None,
) -> np.ndarray:
    metadata = read_video_metadata(video_path)
    safe_frame_index = min(max(0, int(frame_index)), max(0, metadata.frame_count - 1))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise PreprocessError(f"No se pudo abrir el video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, safe_frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise PreprocessError(f"No se pudo leer el frame {safe_frame_index} de preview: {video_path}")

    transformed = _rotate_frame(frame, rotation)
    validated_roi = validate_roi(roi, transformed.shape[1], transformed.shape[0])
    preview = _draw_roi(transformed, validated_roi)
    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
