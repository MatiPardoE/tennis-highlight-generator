from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from config import DetectionConfig, PipelineManifest, Point, PreprocessConfig, ROI, RenderConfig
from detect_segments import DetectionError, detect_useful_segments
from main import build_default_workspace
from preprocess import (
    PreprocessError,
    generate_proxy,
    read_preview_frame,
    read_video_metadata,
    validate_roi,
)
from render_from_segments import RenderError, render_highlights

DEFAULT_ROI_REFERENCE_SIZE = (1846, 922)
DEFAULT_ROI_POINTS = (
    Point(800, 300),
    Point(1200, 300),
    Point(1845, 540),
    Point(1845, 921),
    Point(0, 921),
    Point(0, 540),
)


def _default_workspace_bundle(raw_original_path: str) -> tuple[str, str, str]:
    if raw_original_path.strip():
        workspace = build_default_workspace(Path(raw_original_path.strip())).resolve()
    else:
        workspace = (Path.cwd() / "runs" / "session").resolve()
    manifest = (workspace / "segments_manifest.json").resolve()
    output = (workspace / "highlights.mp4").resolve()
    return str(workspace), str(manifest), str(output)


def _ensure_path_state() -> None:
    state = st.session_state
    if "original_video_raw" not in state:
        state["original_video_raw"] = ""

    current_original = str(state.get("original_video_raw", ""))
    initialized_for = state.get("paths_initialized_for")

    if (
        "workspace_raw" not in state
        or "manifest_raw" not in state
        or "output_raw" not in state
        or initialized_for != current_original
    ):
        workspace, manifest, output = _default_workspace_bundle(current_original)
        state["workspace_raw"] = workspace
        state["manifest_raw"] = manifest
        state["output_raw"] = output
        state["paths_initialized_for"] = current_original


def _load_manifest_if_exists(manifest_path: Path) -> PipelineManifest | None:
    if not manifest_path.exists():
        return None
    return PipelineManifest.load(manifest_path)


def _segments_rows(manifest: PipelineManifest) -> list[dict[str, float]]:
    return [
        {
            "start_sec": round(segment.start_sec, 3),
            "end_sec": round(segment.end_sec, 3),
            "duration_sec": round(segment.end_sec - segment.start_sec, 3),
        }
        for segment in manifest.useful_segments
    ]


def _overlay_roi(frame_rgb: np.ndarray, roi: ROI | None) -> np.ndarray:
    if roi is None:
        return frame_rgb
    preview = frame_rgb.copy()
    polygon = np.array([[point.x, point.y] for point in roi.points], dtype=np.int32)
    cv2.polylines(preview, [polygon], isClosed=True, color=(0, 255, 0), thickness=3)
    for index, point in enumerate(roi.points, start=1):
        cv2.circle(preview, (point.x, point.y), 7, (255, 128, 0), -1)
        cv2.putText(
            preview,
            str(index),
            (point.x + 8, point.y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 128, 0),
            2,
            cv2.LINE_AA,
        )
    return preview


def _roi_state_prefix(video_key: str) -> str:
    return f"roi::{video_key}"


def _get_roi_from_state(video_key: str) -> ROI | None:
    prefix = _roi_state_prefix(video_key)
    if f"{prefix}::enabled" not in st.session_state or not st.session_state[f"{prefix}::enabled"]:
        return None

    return ROI(
        points=tuple(
            Point(
                x=int(st.session_state.get(f"{prefix}::p{index}x", 0)),
                y=int(st.session_state.get(f"{prefix}::p{index}y", 0)),
            )
            for index in range(1, 7)
        ),
    )


def _set_roi_in_state(video_key: str, roi: ROI | None) -> None:
    prefix = _roi_state_prefix(video_key)
    if roi is None:
        for index in range(1, 7):
            st.session_state[f"{prefix}::p{index}x"] = 0
            st.session_state[f"{prefix}::p{index}y"] = 0
        return

    for index, point in enumerate(roi.points, start=1):
        st.session_state[f"{prefix}::p{index}x"] = int(point.x)
        st.session_state[f"{prefix}::p{index}y"] = int(point.y)


def _default_roi(frame_width: int, frame_height: int) -> ROI:
    reference_width, reference_height = DEFAULT_ROI_REFERENCE_SIZE
    max_x = max(0, frame_width - 1)
    max_y = max(0, frame_height - 1)

    scaled_points = []
    for point in DEFAULT_ROI_POINTS:
        scaled_x = round((point.x / max(1, reference_width - 1)) * max_x)
        scaled_y = round((point.y / max(1, reference_height - 1)) * max_y)
        scaled_points.append(
            Point(
                x=max(0, min(int(scaled_x), max_x)),
                y=max(0, min(int(scaled_y), max_y)),
            )
        )

    return ROI(points=tuple(scaled_points))


def _clip_roi_to_frame(roi: ROI, frame_width: int, frame_height: int) -> ROI:
    return ROI(
        points=tuple(
            Point(
                x=max(0, min(int(point.x), frame_width - 1)),
                y=max(0, min(int(point.y), frame_height - 1)),
            )
            for point in roi.points
        )
    )


def _apply_point_action(
    roi: ROI,
    action: str,
    step: int,
    selected_point_idx: int,
    frame_width: int,
    frame_height: int,
) -> ROI:
    points = [Point(point.x, point.y) for point in roi.points]
    selected_point = points[selected_point_idx]

    if action == "move_left":
        selected_point.x -= step
    elif action == "move_right":
        selected_point.x += step
    elif action == "move_up":
        selected_point.y -= step
    elif action == "move_down":
        selected_point.y += step

    points[selected_point_idx] = selected_point
    return _clip_roi_to_frame(ROI(points=tuple(points)), frame_width, frame_height)


def _ensure_roi_state(video_key: str, frame_width: int, frame_height: int) -> None:
    prefix = _roi_state_prefix(video_key)
    if f"{prefix}::enabled" not in st.session_state:
        st.session_state[f"{prefix}::enabled"] = False
    if f"{prefix}::selected_point" not in st.session_state:
        st.session_state[f"{prefix}::selected_point"] = 1
    if f"{prefix}::p1x" not in st.session_state:
        _set_roi_in_state(video_key, _default_roi(frame_width, frame_height))

    current_roi = _get_roi_from_state(video_key)
    if current_roi is None:
        return

    _set_roi_in_state(video_key, _clip_roi_to_frame(current_roi, frame_width, frame_height))


def _render_roi_editor(video_key: str, calibration_frame_rgb: np.ndarray) -> ROI | None:
    frame_height, frame_width = calibration_frame_rgb.shape[:2]
    _ensure_roi_state(video_key, frame_width, frame_height)
    prefix = _roi_state_prefix(video_key)

    pending_action = st.session_state.pop(f"{prefix}::pending_action", None)
    if pending_action == "fullframe":
        st.session_state[f"{prefix}::enabled"] = True
        _set_roi_in_state(video_key, _default_roi(frame_width, frame_height))
    elif pending_action == "clear":
        st.session_state[f"{prefix}::enabled"] = False
        _set_roi_in_state(video_key, None)
    elif pending_action in {"move_left", "move_right", "move_up", "move_down"}:
        base_roi = _get_roi_from_state(video_key) or _default_roi(frame_width, frame_height)
        step = max(1, int(st.session_state.get(f"{prefix}::step", 10)))
        st.session_state[f"{prefix}::enabled"] = True
        _set_roi_in_state(
            video_key,
            _apply_point_action(
                roi=base_roi,
                action=pending_action,
                step=step,
                selected_point_idx=max(0, int(st.session_state.get(f"{prefix}::selected_point", 1)) - 1),
                frame_width=frame_width,
                frame_height=frame_height,
            ),
        )

    st.subheader("Calibración de ROI")
    enabled = st.checkbox(
        "Aplicar ROI adicional después de la rotación",
        key=f"{prefix}::enabled",
    )
    if not enabled:
        return None

    current_roi = _get_roi_from_state(video_key)
    if current_roi is None:
        current_roi = _default_roi(frame_width, frame_height)
        _set_roi_in_state(video_key, current_roi)

    left_col, right_col = st.columns([1.8, 1.2])

    with left_col:
        st.caption("Preview del ROI poligonal de 6 puntos sobre el frame calibrado.")
        st.image(_overlay_roi(calibration_frame_rgb, current_roi), use_container_width=True)

        st.caption("Punto activo")
        point_cols_top = st.columns(3)
        point_cols_bottom = st.columns(3)
        point_buttons = point_cols_top + point_cols_bottom
        for index, label in enumerate(["P1", "P2", "P3", "P4", "P5", "P6"], start=1):
            if point_buttons[index - 1].button(
                label,
                key=f"{prefix}::select_p{index}",
                use_container_width=True,
                type="primary" if int(st.session_state.get(f"{prefix}::selected_point", 1)) == index else "secondary",
            ):
                st.session_state[f"{prefix}::selected_point"] = index
                st.rerun()

        st.caption("Mover punto activo")
        move_cols = st.columns(4)
        if move_cols[0].button("←", key=f"{prefix}::move_left", use_container_width=True):
            st.session_state[f"{prefix}::pending_action"] = "move_left"
            st.rerun()
        if move_cols[1].button("→", key=f"{prefix}::move_right", use_container_width=True):
            st.session_state[f"{prefix}::pending_action"] = "move_right"
            st.rerun()
        if move_cols[2].button("↑", key=f"{prefix}::move_up", use_container_width=True):
            st.session_state[f"{prefix}::pending_action"] = "move_up"
            st.rerun()
        if move_cols[3].button("↓", key=f"{prefix}::move_down", use_container_width=True):
            st.session_state[f"{prefix}::pending_action"] = "move_down"
            st.rerun()

        if st.button("Usar frame completo", key=f"{prefix}::fullframe", use_container_width=True):
            st.session_state[f"{prefix}::pending_action"] = "fullframe"
            st.rerun()

        if st.button("Limpiar ROI", key=f"{prefix}::clear", use_container_width=True):
            st.session_state[f"{prefix}::pending_action"] = "clear"
            st.rerun()

    with right_col:
        st.caption("Ajuste fino")
        st.caption(
            "Orden recomendado en sentido horario: "
            "P1 arriba-izquierda, P2 arriba-centro, P3 arriba-derecha, "
            "P4 abajo-derecha, P5 abajo-centro, P6 abajo-izquierda."
        )
        st.number_input(
            "Paso de ajuste (px)",
            min_value=1,
            max_value=max(frame_width, frame_height),
            value=int(st.session_state.get(f"{prefix}::step", 10)),
            key=f"{prefix}::step",
        )
        point_rows = []
        for index in range(1, 7):
            px = st.number_input(
                f"P{index} x",
                min_value=0,
                max_value=max(0, frame_width - 1),
                value=int(st.session_state.get(f"{prefix}::p{index}x", 0)),
                key=f"{prefix}::p{index}x",
            )
            py = st.number_input(
                f"P{index} y",
                min_value=0,
                max_value=max(0, frame_height - 1),
                value=int(st.session_state.get(f"{prefix}::p{index}y", 0)),
                key=f"{prefix}::p{index}y",
            )
            point_rows.append(Point(x=int(px), y=int(py)))

    validated_roi = validate_roi(ROI(points=tuple(point_rows)), frame_width, frame_height)

    st.caption(
        "ROI final: "
        + ", ".join(
            f"P{index}=({point.x},{point.y})"
            for index, point in enumerate(validated_roi.points, start=1)
        )
    )
    return validated_roi


def main() -> None:
    st.set_page_config(page_title="Tennis Highlight Generator", layout="wide")
    st.title("Tennis Highlight Generator")
    st.caption("Pipeline local proxy -> detección -> render final sobre el video original.")
    _ensure_path_state()

    with st.sidebar:
        st.subheader("Fuente local")
        original_video_raw = st.text_input("Video original", key="original_video_raw")
        workspace_raw = st.text_input("Workspace", key="workspace_raw")
        manifest_raw = st.text_input("Manifest JSON", key="manifest_raw")
        output_raw = st.text_input("Video final", key="output_raw")

        st.subheader("Preprocesado")
        rotation = st.number_input("Ángulo de rotación (grados)", min_value=-180.0, max_value=180.0, value=0.0, step=0.1)
        proxy_fps = st.slider("FPS proxy", min_value=1.0, max_value=12.0, value=8.0, step=1.0)
        proxy_width = st.slider("Ancho proxy", min_value=240, max_value=1280, value=640, step=80)
        proxy_crf = st.slider("CRF proxy", min_value=20, max_value=38, value=34, step=1)
        proxy_preset = st.selectbox(
            "Preset proxy",
            options=["ultrafast", "veryfast", "faster", "fast", "medium"],
            index=0,
        )

        st.subheader("Detección")
        sensitivity = st.slider("Sensibilidad", min_value=0.0, max_value=1.0, value=0.55, step=0.05)
        smooth_window_sec = st.slider("Suavizado (s)", min_value=0.2, max_value=3.0, value=1.0, step=0.1)
        min_segment_sec = st.slider("Duración mínima útil (s)", min_value=1.0, max_value=20.0, value=4.0, step=0.5)
        min_gap_sec = st.slider("Gap máximo a mergear (s)", min_value=0.0, max_value=10.0, value=2.0, step=0.25)
        padding_sec = st.slider("Padding (s)", min_value=0.0, max_value=2.0, value=0.35, step=0.05)
        blur_kernel_size = st.slider("Blur kernel", min_value=1, max_value=11, value=5, step=2)
        yolo_model = st.text_input("Modelo YOLO", value="yolov8n.pt")
        person_confidence = st.slider("Confianza persona", min_value=0.10, max_value=0.90, value=0.30, step=0.05)
        flow_metric = st.selectbox("Métrica flow", options=["p90", "mean", "fast_ratio"], index=0)
        flow_fast_threshold = st.slider("Umbral flow rápido", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
        global_activity_mode = st.selectbox("Combinación score jugadores", options=["max", "mean"], index=0)

        st.subheader("Render")
        render_crf = st.slider("CRF final", min_value=16, max_value=28, value=18, step=1)
        render_preset = st.selectbox(
            "Preset final",
            options=["fast", "medium", "slow"],
            index=1,
        )

    original_video_path = Path(original_video_raw).expanduser() if original_video_raw.strip() else None
    workspace_path = Path(workspace_raw).expanduser().resolve()
    manifest_path = Path(manifest_raw).expanduser().resolve()
    output_path = Path(output_raw).expanduser().resolve()

    preview_roi: ROI | None = None
    calibration_frame_rgb: np.ndarray | None = None
    calibration_sec = 0.0

    if original_video_path is not None and original_video_path.exists():
        try:
            metadata = read_video_metadata(original_video_path)
            calibration_sec = st.slider(
                "Inicio útil / frame para calibración",
                min_value=0.0,
                max_value=max(0.0, float(metadata.duration_sec)),
                value=min(3.0, float(metadata.duration_sec)),
                step=max(1.0 / max(metadata.fps, 1.0), 0.1),
            )
            calibration_frame_index = min(
                max(0, int(round(calibration_sec * metadata.fps))),
                max(0, metadata.frame_count - 1),
            )
            st.caption(
                f"Usando el frame {calibration_frame_index} "
                f"({calibration_frame_index / metadata.fps:.2f}s) para definir ángulo y ROI. "
                f"El proxy y la detección arrancarán desde ese punto."
            )

            calibration_frame_rgb = read_preview_frame(
                original_video_path,
                frame_index=calibration_frame_index,
                rotation=rotation,
                roi=None,
            )
            st.subheader("Frame calibrado")
            st.caption(
                f"Vista ya rotada y autocropeada para evitar bordes negros. "
                f"Tamaño útil: {calibration_frame_rgb.shape[1]}x{calibration_frame_rgb.shape[0]}."
            )

            video_key = f"{original_video_path.resolve()}::{round(rotation, 3)}"
            preview_roi = _render_roi_editor(video_key, calibration_frame_rgb)
            final_preview = _overlay_roi(calibration_frame_rgb, preview_roi)
            st.image(final_preview, caption="Preview final con ROI aplicado")
        except PreprocessError as exc:
            st.error(str(exc))
            metadata = None
    elif original_video_raw.strip():
        st.warning("El path del video original no existe.")
        metadata = None
    else:
        metadata = None

    preprocess_cfg = PreprocessConfig(
        rotation=float(rotation),
        roi=preview_roi,
        analysis_start_sec=float(calibration_sec),
        proxy_fps=proxy_fps,
        proxy_width=proxy_width,
        proxy_crf=proxy_crf,
        proxy_preset=proxy_preset,
    )
    detection_cfg = DetectionConfig(
        sensitivity=sensitivity,
        smooth_window_sec=smooth_window_sec,
        min_segment_sec=min_segment_sec,
        min_gap_sec=min_gap_sec,
        padding_sec=padding_sec,
        blur_kernel_size=blur_kernel_size,
        yolo_model=yolo_model.strip() or "yolov8n.pt",
        person_confidence=person_confidence,
        flow_metric=flow_metric,
        flow_fast_threshold=flow_fast_threshold,
        global_activity_mode=global_activity_mode,
    )
    render_cfg = RenderConfig(
        crf=render_crf,
        preset=render_preset,
    )

    col1, col2, col3, col4 = st.columns(4)
    run_preprocess = col1.button("1. Generar proxy", use_container_width=True)
    run_detect = col2.button("2. Detectar", use_container_width=True)
    run_render = col3.button("3. Renderizar", use_container_width=True)
    run_all = col4.button("Pipeline completo", type="primary", use_container_width=True)

    try:
        if run_preprocess or run_all:
            if original_video_path is None:
                st.error("Indicá un video original local.")
                return
            progress_box = st.empty()
            progress_bar = st.progress(0)

            def on_proxy_progress(message: str, progress: float | None) -> None:
                progress_box.info(message)
                if progress is not None:
                    progress_bar.progress(max(0, min(int(progress * 100), 100)))

            with st.spinner("Generando proxy..."):
                manifest = generate_proxy(
                    original_video_path=original_video_path,
                    workspace_dir=workspace_path,
                    preprocess_cfg=preprocess_cfg,
                    manifest_path=manifest_path,
                    progress_callback=on_proxy_progress,
                )
            progress_bar.progress(100)
            progress_box.success(f"Proxy generado en {manifest.proxy_video_path}")
            st.success(f"Proxy generado en {manifest.proxy_video_path}")

        if run_detect or run_all:
            if not manifest_path.exists():
                st.error("No existe el manifest JSON. Generá primero el proxy.")
                return
            with st.spinner("Detectando segmentos..."):
                manifest = detect_useful_segments(
                    manifest_path=manifest_path,
                    detection_cfg=detection_cfg,
                )
            st.success(f"Segmentos detectados: {len(manifest.useful_segments)}")

        if run_render or run_all:
            if not manifest_path.exists():
                st.error("No existe el manifest JSON. Ejecutá preprocess y detect antes de renderizar.")
                return
            with st.spinner("Renderizando highlights..."):
                render_highlights(
                    manifest_path=manifest_path,
                    output_path=output_path,
                    render_cfg=render_cfg,
                )
            st.success(f"Highlights exportados en {output_path}")
    except (ValueError, PreprocessError, DetectionError, RenderError) as exc:
        st.error(str(exc))

    manifest = _load_manifest_if_exists(manifest_path)
    if manifest is None:
        st.info("Todavía no existe un manifest. Ejecutá el preprocesado para crearlo.")
        return

    metrics = st.columns(4)
    metrics[0].metric("FPS original", f"{manifest.fps_original:.2f}")
    metrics[1].metric("FPS proxy", f"{manifest.fps_proxy:.2f}")
    metrics[2].metric("Segmentos útiles", str(len(manifest.useful_segments)))
    metrics[3].metric("Inicio útil", f"{manifest.analysis_start_sec:.1f}s")

    st.subheader("Manifest JSON")
    st.code(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False), language="json")

    if manifest.useful_segments:
        st.subheader("Segmentos útiles")
        st.dataframe(_segments_rows(manifest), use_container_width=True, hide_index=True)

    if manifest.debug_video_path:
        debug_video_path = Path(manifest.debug_video_path)
        if debug_video_path.exists():
            st.subheader("Debug detección")
            st.caption(f"Guardado en: {debug_video_path}")
            st.video(str(debug_video_path))

    if output_path.exists():
        st.subheader("Highlights")
        st.video(str(output_path))


if __name__ == "__main__":
    main()
