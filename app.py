from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.config import ExportConfig, ProcessingConfig
from src.debug_export import DebugExportError, export_player_activity_debug_video
from src.export import ExportError, export_highlight_video
from src.logging_utils import close_run_logger, create_run_logger
from src.motion_features import MotionExtractionError, extract_motion_series
from src.postprocess import build_non_game_segments, postprocess_game_segments
from src.profiling import StageProfiler
from src.preview import format_seconds, segments_to_rows, total_duration
from src.segment_detection import detect_initial_segments
from src.utils import create_workspace, sanitize_stem
from src.video_io import VideoIOError, ensure_ffmpeg_available, read_video_metadata, save_uploaded_video


def run_pipeline(uploaded_file, cfg: ProcessingConfig, export_cfg: ExportConfig) -> dict:
    workspace = create_workspace()
    logger, log_path = create_run_logger(workspace)
    profiler = StageProfiler()

    try:
        uploaded_bytes = uploaded_file.getvalue()
        logger.info(
            "Inicio pipeline. file='%s' size_mb=%.2f",
            uploaded_file.name,
            len(uploaded_bytes) / (1024.0 * 1024.0),
        )

        with profiler.track("save_upload"):
            input_path = save_uploaded_video(uploaded_bytes, uploaded_file.name, workspace)

        with profiler.track("ensure_ffmpeg"):
            ensure_ffmpeg_available()
        with profiler.track("read_metadata"):
            metadata = read_video_metadata(input_path)
        with profiler.track("extract_motion"):
            motion = extract_motion_series(input_path, metadata, cfg, logger=logger)

        with profiler.track("detect_initial_segments"):
            threshold, initial_segments = detect_initial_segments(motion, cfg)
        with profiler.track("postprocess_segments"):
            final_segments = postprocess_game_segments(initial_segments, cfg, metadata.duration_sec)
            non_game_segments = build_non_game_segments(final_segments, metadata.duration_sec)

        output_path: Path | None = None
        if final_segments:
            with profiler.track("export_highlights"):
                output_path = workspace / f"{sanitize_stem(uploaded_file.name)}_highlights.mp4"
                export_highlight_video(input_path, output_path, final_segments, export_cfg)

        debug_video_path: Path | None = None
        if cfg.debug_mode:
            with profiler.track("export_debug_video"):
                debug_video_path = workspace / f"{sanitize_stem(uploaded_file.name)}_debug_player_activity.mp4"
                export_player_activity_debug_video(input_path, debug_video_path, metadata, motion, cfg, threshold)

        pipeline_profile = dict(profiler.stage_seconds)
        pipeline_profile["total"] = profiler.total_seconds
        profiling = {
            "pipeline": pipeline_profile,
            "motion": motion.profiling,
        }

        profiling_path = workspace / "profiling.json"
        profiling_path.write_text(json.dumps(profiling, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info("Pipeline finalizado en %.2fs. workspace=%s", profiler.total_seconds, workspace)
        for stage_name, elapsed in sorted(pipeline_profile.items(), key=lambda item: item[1], reverse=True):
            logger.info("Stage %-24s %.2fs", stage_name, elapsed)

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
            "debug_video_path": debug_video_path,
            "log_path": log_path,
            "profiling_path": profiling_path,
            "profiling": profiling,
        }
    except Exception:
        logger.exception("Pipeline falló.")
        raise
    finally:
        close_run_logger(logger)


def main() -> None:
    st.set_page_config(page_title="Tennis Highlight Generator", layout="wide")
    st.title("Tennis Highlight Generator (MVP)")
    st.caption("Recorta videos de tenis amateur eliminando pausas con heurística de jugadores + optical flow local.")

    with st.sidebar:
        st.subheader("Parámetros de segmentación")
        sensitivity = st.slider("Sensibilidad", min_value=0.0, max_value=1.0, value=0.55, step=0.05)
        min_game_sec = st.slider("Duración mínima de juego (s)", min_value=1.0, max_value=20.0, value=4.0, step=0.5)
        min_pause_sec = st.slider("Duración mínima de pausa (s)", min_value=0.0, max_value=10.0, value=2.0, step=0.25)
        padding_sec = st.slider("Padding antes/después (s)", min_value=0.0, max_value=2.0, value=0.35, step=0.05)
        sample_fps = st.slider("Muestreo de frames (fps)", min_value=1.0, max_value=12.0, value=5.0, step=1.0)
        smooth_window_sec = st.slider("Suavizado temporal (s)", min_value=0.2, max_value=3.0, value=1.0, step=0.1)
        st.subheader("Jugadores + Optical Flow")
        yolo_model = st.text_input("Modelo YOLO", value="yolov8n.pt")
        person_confidence = st.slider("Confianza mínima persona", min_value=0.10, max_value=0.90, value=0.35, step=0.05)
        flow_metric = st.selectbox(
            "Métrica local de flow",
            options=["p90", "mean", "fast_ratio"],
            index=0,
        )
        flow_fast_threshold = st.slider(
            "Umbral de flow rápido (px/frame)",
            min_value=0.5,
            max_value=8.0,
            value=2.0,
            step=0.1,
        )
        global_activity_mode = st.selectbox(
            "Combinación de actividad global",
            options=["max", "mean"],
            index=0,
        )
        debug_mode = st.checkbox("Modo debug", value=False)

    cfg = ProcessingConfig(
        sensitivity=sensitivity,
        min_game_sec=min_game_sec,
        min_pause_sec=min_pause_sec,
        padding_sec=padding_sec,
        sample_fps=sample_fps,
        smooth_window_sec=smooth_window_sec,
        yolo_model=yolo_model.strip() or "yolov8n.pt",
        person_confidence=person_confidence,
        flow_metric=flow_metric,
        flow_fast_threshold=flow_fast_threshold,
        global_activity_mode=global_activity_mode,
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
    except (VideoIOError, MotionExtractionError, ExportError, DebugExportError) as exc:
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
    debug_video_path: Path | None = result["debug_video_path"]
    log_path: Path = result["log_path"]
    profiling_path: Path = result["profiling_path"]
    profiling: dict = result["profiling"]

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
        st.dataframe(segments_to_rows(final_segments), width='stretch', hide_index=True)
    else:
        st.warning("No se detectaron segmentos GAME con los parámetros actuales.")

    st.subheader("Profiling")
    stage_rows = [
        {"stage": stage, "seconds": round(seconds, 3)}
        for stage, seconds in sorted(profiling["pipeline"].items(), key=lambda item: item[1], reverse=True)
    ]
    st.dataframe(stage_rows, width='stretch', hide_index=True)
    motion_profile = profiling.get("motion", {})
    if motion_profile:
        st.write(
            "Motion extractor:",
            f"pares={int(motion_profile.get('sampled_pairs', 0))}",
            f"throughput={motion_profile.get('throughput_pairs_per_sec', 0.0):.2f} pares/s",
            f"yolo={motion_profile.get('detection_seconds', 0.0):.2f}s",
            f"flow={motion_profile.get('flow_seconds', 0.0):.2f}s",
        )
    st.caption(f"Log: {log_path}")
    st.caption(f"Profiling JSON: {profiling_path}")
    st.download_button(
        label="Descargar log de pipeline",
        data=log_path.read_bytes(),
        file_name=log_path.name,
        mime="text/plain",
    )
    st.download_button(
        label="Descargar profiling JSON",
        data=profiling_path.read_bytes(),
        file_name=profiling_path.name,
        mime="application/json",
    )

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
        if debug_video_path is not None:
            st.write("Video debug: cajas de jugadores y score de optical flow por frame")
            st.video(str(debug_video_path))
            st.download_button(
                label="Descargar video debug de actividad",
                data=debug_video_path.read_bytes(),
                file_name=debug_video_path.name,
                mime="video/mp4",
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
