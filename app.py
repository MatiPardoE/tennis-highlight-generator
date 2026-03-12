from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.config import ExportConfig, ProcessingConfig
from src.export import ExportError, export_highlight_video
from src.motion_features import MotionExtractionError, extract_motion_series
from src.postprocess import build_non_game_segments, postprocess_game_segments
from src.preview import format_seconds, segments_to_rows, total_duration
from src.segment_detection import detect_initial_segments
from src.utils import create_workspace, sanitize_stem
from src.video_io import VideoIOError, ensure_ffmpeg_available, read_video_metadata, save_uploaded_video


def run_pipeline(uploaded_file, cfg: ProcessingConfig, export_cfg: ExportConfig) -> dict:
    workspace = create_workspace()
    input_path = save_uploaded_video(uploaded_file.getvalue(), uploaded_file.name, workspace)

    ensure_ffmpeg_available()
    metadata = read_video_metadata(input_path)
    motion = extract_motion_series(input_path, metadata, cfg)

    threshold, initial_segments = detect_initial_segments(motion, cfg)
    final_segments = postprocess_game_segments(initial_segments, cfg, metadata.duration_sec)
    non_game_segments = build_non_game_segments(final_segments, metadata.duration_sec)

    output_path: Path | None = None
    if final_segments:
        output_path = workspace / f"{sanitize_stem(uploaded_file.name)}_highlights.mp4"
        export_highlight_video(input_path, output_path, final_segments, export_cfg)

    return {
        "workspace": workspace,
        "input_path": input_path,
        "metadata": metadata,
        "motion": motion,
        "threshold": threshold,
        "initial_segments": initial_segments,
        "final_segments": final_segments,
        "non_game_segments": non_game_segments,
        "output_path": output_path,
    }


def main() -> None:
    st.set_page_config(page_title="Tennis Highlight Generator", layout="wide")
    st.title("Tennis Highlight Generator (MVP)")
    st.caption("Recorta videos de tenis amateur eliminando pausas con heurísticas de movimiento.")

    with st.sidebar:
        st.subheader("Parámetros")
        sensitivity = st.slider("Sensibilidad", min_value=0.0, max_value=1.0, value=0.55, step=0.05)
        min_game_sec = st.slider("Duración mínima de juego (s)", min_value=1.0, max_value=20.0, value=4.0, step=0.5)
        min_pause_sec = st.slider("Duración mínima de pausa (s)", min_value=0.0, max_value=10.0, value=2.0, step=0.25)
        padding_sec = st.slider("Padding antes/después (s)", min_value=0.0, max_value=2.0, value=0.35, step=0.05)
        sample_fps = st.slider("Muestreo de frames (fps)", min_value=1.0, max_value=12.0, value=5.0, step=1.0)
        smooth_window_sec = st.slider("Suavizado temporal (s)", min_value=0.2, max_value=3.0, value=1.0, step=0.1)
        debug_mode = st.checkbox("Modo debug", value=False)

    cfg = ProcessingConfig(
        sensitivity=sensitivity,
        min_game_sec=min_game_sec,
        min_pause_sec=min_pause_sec,
        padding_sec=padding_sec,
        sample_fps=sample_fps,
        smooth_window_sec=smooth_window_sec,
        debug_mode=debug_mode,
    )
    export_cfg = ExportConfig()

    uploaded_file = st.file_uploader(
        "Subí un video de tenis",
        type=["mp4", "mov", "avi", "mkv", "m4v"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Cargá un video para empezar.")
        return

    st.video(uploaded_file)

    if not st.button("Procesar video", type="primary"):
        return

    try:
        with st.spinner("Procesando video..."):
            result = run_pipeline(uploaded_file, cfg, export_cfg)
    except (VideoIOError, MotionExtractionError, ExportError) as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Error inesperado: {exc}")
        return

    metadata = result["metadata"]
    initial_segments = result["initial_segments"]
    final_segments = result["final_segments"]
    non_game_segments = result["non_game_segments"]
    output_path: Path | None = result["output_path"]

    kept_sec = total_duration(final_segments)
    removed_ratio = 0.0
    if metadata.duration_sec > 0:
        removed_ratio = max(0.0, 1.0 - (kept_sec / metadata.duration_sec))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Duración original", format_seconds(metadata.duration_sec))
    col2.metric("Duración GAME", format_seconds(kept_sec))
    col3.metric("Reducción", f"{removed_ratio * 100:.1f}%")
    col4.metric("Segmentos GAME", str(len(final_segments)))

    st.subheader("Resumen de segmentos")
    st.write(f"Segmentos GAME iniciales: {len(initial_segments)}")
    st.write(f"Segmentos GAME finales: {len(final_segments)}")
    st.write(f"Segmentos NON_GAME: {len(non_game_segments)}")

    if final_segments:
        st.dataframe(segments_to_rows(final_segments), use_container_width=True, hide_index=True)
    else:
        st.warning("No se detectaron segmentos GAME con los parámetros actuales.")

    if cfg.debug_mode:
        motion = result["motion"]
        threshold = result["threshold"]
        st.subheader("Debug")
        st.write(f"Umbral de actividad: {threshold:.4f}")
        st.line_chart(
            {
                "score": motion.scores,
                "score_suavizado": motion.smoothed_scores,
                "umbral": [threshold] * len(motion.scores),
            }
        )

    if output_path is None:
        return

    st.subheader("Video exportado")
    st.video(str(output_path))

    st.download_button(
        label="Descargar highlights",
        data=output_path.read_bytes(),
        file_name=output_path.name,
        mime="video/mp4",
    )


if __name__ == "__main__":
    main()
