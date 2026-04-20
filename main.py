from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from config import DetectionConfig, PreprocessConfig, RenderConfig, parse_roi
from detect_segments import DetectionError, detect_useful_segments
from preprocess import PreprocessError, generate_proxy
from render_from_segments import RenderError, render_highlights


def build_default_workspace(input_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path.cwd() / "runs" / f"{input_path.stem}_{timestamp}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline local para proxy, detección de segmentos y render final sobre el original.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser("preprocess", help="Genera proxy y manifiesto JSON.")
    preprocess_parser.add_argument("--input", required=True, help="Path al video original.")
    preprocess_parser.add_argument("--workspace", help="Directorio de salida.")
    preprocess_parser.add_argument("--manifest", help="Path al manifiesto JSON.")
    preprocess_parser.add_argument("--proxy-name", default="proxy.mp4", help="Nombre del archivo proxy.")
    preprocess_parser.add_argument("--rotation", type=float, default=0.0, help="Ángulo en grados. Acepta decimales.")
    preprocess_parser.add_argument("--roi", help="ROI poligonal de 6 puntos en formato x1,y1;x2,y2;x3,y3;x4,y4;x5,y5;x6,y6.")
    preprocess_parser.add_argument("--analysis-start-sec", type=float, default=0.0, help="Segundo desde el que arranca el proxy y la detección.")
    preprocess_parser.add_argument("--proxy-fps", type=float, default=8.0)
    preprocess_parser.add_argument("--proxy-width", type=int, default=640)
    preprocess_parser.add_argument("--proxy-crf", type=int, default=34)
    preprocess_parser.add_argument("--proxy-preset", default="ultrafast")

    detect_parser = subparsers.add_parser("detect", help="Detecta segmentos útiles sobre el proxy.")
    detect_parser.add_argument("--manifest", required=True, help="Path al manifiesto JSON.")
    detect_parser.add_argument("--output-manifest", help="Path de salida para el JSON actualizado.")
    detect_parser.add_argument("--sensitivity", type=float, default=0.55)
    detect_parser.add_argument("--smooth-window-sec", type=float, default=1.0)
    detect_parser.add_argument("--min-segment-sec", type=float, default=4.0)
    detect_parser.add_argument("--min-gap-sec", type=float, default=2.0)
    detect_parser.add_argument("--padding-sec", type=float, default=0.35)
    detect_parser.add_argument("--blur-kernel-size", type=int, default=5)
    detect_parser.add_argument("--yolo-model", default="yolov8n.pt")
    detect_parser.add_argument("--person-confidence", type=float, default=0.30)
    detect_parser.add_argument("--flow-metric", default="p90")
    detect_parser.add_argument("--flow-fast-threshold", type=float, default=2.0)
    detect_parser.add_argument("--global-activity-mode", default="max")

    render_parser = subparsers.add_parser("render", help="Exporta highlights desde el original.")
    render_parser.add_argument("--manifest", required=True, help="Path al manifiesto JSON.")
    render_parser.add_argument("--output", required=True, help="Path del video final.")
    render_parser.add_argument("--crf", type=int, default=18)
    render_parser.add_argument("--preset", default="medium")

    run_all_parser = subparsers.add_parser("run-all", help="Ejecuta preprocess, detect y render.")
    run_all_parser.add_argument("--input", required=True, help="Path al video original.")
    run_all_parser.add_argument("--workspace", help="Directorio de salida.")
    run_all_parser.add_argument("--manifest", help="Path al manifiesto JSON.")
    run_all_parser.add_argument("--output", help="Path del video final.")
    run_all_parser.add_argument("--proxy-name", default="proxy.mp4")
    run_all_parser.add_argument("--rotation", type=float, default=0.0)
    run_all_parser.add_argument("--roi", help="ROI poligonal de 6 puntos en formato x1,y1;x2,y2;x3,y3;x4,y4;x5,y5;x6,y6.")
    run_all_parser.add_argument("--analysis-start-sec", type=float, default=0.0, help="Segundo desde el que arranca el proxy y la detección.")
    run_all_parser.add_argument("--proxy-fps", type=float, default=8.0)
    run_all_parser.add_argument("--proxy-width", type=int, default=640)
    run_all_parser.add_argument("--proxy-crf", type=int, default=34)
    run_all_parser.add_argument("--proxy-preset", default="ultrafast")
    run_all_parser.add_argument("--sensitivity", type=float, default=0.55)
    run_all_parser.add_argument("--smooth-window-sec", type=float, default=1.0)
    run_all_parser.add_argument("--min-segment-sec", type=float, default=4.0)
    run_all_parser.add_argument("--min-gap-sec", type=float, default=2.0)
    run_all_parser.add_argument("--padding-sec", type=float, default=0.35)
    run_all_parser.add_argument("--blur-kernel-size", type=int, default=5)
    run_all_parser.add_argument("--yolo-model", default="yolov8n.pt")
    run_all_parser.add_argument("--person-confidence", type=float, default=0.30)
    run_all_parser.add_argument("--flow-metric", default="p90")
    run_all_parser.add_argument("--flow-fast-threshold", type=float, default=2.0)
    run_all_parser.add_argument("--global-activity-mode", default="max")
    run_all_parser.add_argument("--crf", type=int, default=18)
    run_all_parser.add_argument("--preset", default="medium")

    return parser


def _resolve_workspace(raw_workspace: str | None, input_path: Path) -> Path:
    if raw_workspace:
        return Path(raw_workspace).expanduser().resolve()
    return build_default_workspace(input_path.expanduser().resolve())


def _resolve_manifest(raw_manifest: str | None, workspace: Path) -> Path:
    if raw_manifest:
        return Path(raw_manifest).expanduser().resolve()
    return (workspace / "segments_manifest.json").resolve()


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "preprocess":
            input_path = Path(args.input).expanduser().resolve()
            workspace = _resolve_workspace(args.workspace, input_path)
            manifest_path = _resolve_manifest(args.manifest, workspace)
            manifest = generate_proxy(
                original_video_path=input_path,
                workspace_dir=workspace,
                preprocess_cfg=PreprocessConfig(
                    rotation=args.rotation,
                    roi=parse_roi(args.roi),
                    analysis_start_sec=args.analysis_start_sec,
                    proxy_fps=args.proxy_fps,
                    proxy_width=args.proxy_width,
                    proxy_crf=args.proxy_crf,
                    proxy_preset=args.proxy_preset,
                ),
                manifest_path=manifest_path,
                proxy_filename=args.proxy_name,
            )
            print(f"Proxy generado: {manifest.proxy_video_path}")
            print(f"Manifest JSON: {manifest_path}")
            return 0

        if args.command == "detect":
            manifest_path = Path(args.manifest).expanduser().resolve()
            output_manifest = (
                Path(args.output_manifest).expanduser().resolve()
                if args.output_manifest
                else manifest_path
            )
            manifest = detect_useful_segments(
                manifest_path=manifest_path,
                detection_cfg=DetectionConfig(
                    sensitivity=args.sensitivity,
                    smooth_window_sec=args.smooth_window_sec,
                    min_segment_sec=args.min_segment_sec,
                    min_gap_sec=args.min_gap_sec,
                    padding_sec=args.padding_sec,
                    blur_kernel_size=args.blur_kernel_size,
                    yolo_model=args.yolo_model,
                    person_confidence=args.person_confidence,
                    flow_metric=args.flow_metric,
                    flow_fast_threshold=args.flow_fast_threshold,
                    global_activity_mode=args.global_activity_mode,
                ),
                output_manifest_path=output_manifest,
            )
            print(f"Segmentos detectados: {len(manifest.useful_segments)}")
            print(f"Manifest JSON: {output_manifest}")
            return 0

        if args.command == "render":
            output_path = render_highlights(
                manifest_path=Path(args.manifest).expanduser().resolve(),
                output_path=Path(args.output).expanduser().resolve(),
                render_cfg=RenderConfig(crf=args.crf, preset=args.preset),
            )
            print(f"Highlights exportados: {output_path}")
            return 0

        if args.command == "run-all":
            input_path = Path(args.input).expanduser().resolve()
            workspace = _resolve_workspace(args.workspace, input_path)
            manifest_path = _resolve_manifest(args.manifest, workspace)
            output_path = (
                Path(args.output).expanduser().resolve()
                if args.output
                else (workspace / "highlights.mp4").resolve()
            )

            generate_proxy(
                original_video_path=input_path,
                workspace_dir=workspace,
                preprocess_cfg=PreprocessConfig(
                    rotation=args.rotation,
                    roi=parse_roi(args.roi),
                    analysis_start_sec=args.analysis_start_sec,
                    proxy_fps=args.proxy_fps,
                    proxy_width=args.proxy_width,
                    proxy_crf=args.proxy_crf,
                    proxy_preset=args.proxy_preset,
                ),
                manifest_path=manifest_path,
                proxy_filename=args.proxy_name,
            )
            manifest = detect_useful_segments(
                manifest_path=manifest_path,
                detection_cfg=DetectionConfig(
                    sensitivity=args.sensitivity,
                    smooth_window_sec=args.smooth_window_sec,
                    min_segment_sec=args.min_segment_sec,
                    min_gap_sec=args.min_gap_sec,
                    padding_sec=args.padding_sec,
                    blur_kernel_size=args.blur_kernel_size,
                    yolo_model=args.yolo_model,
                    person_confidence=args.person_confidence,
                    flow_metric=args.flow_metric,
                    flow_fast_threshold=args.flow_fast_threshold,
                    global_activity_mode=args.global_activity_mode,
                ),
            )
            render_highlights(
                manifest_path=manifest_path,
                output_path=output_path,
                render_cfg=RenderConfig(crf=args.crf, preset=args.preset),
            )
            print(f"Proxy generado: {manifest.proxy_video_path}")
            print(f"Manifest JSON: {manifest_path}")
            print(f"Segmentos detectados: {len(manifest.useful_segments)}")
            print(f"Highlights exportados: {output_path}")
            return 0

        parser.error("Comando no soportado.")
    except (ValueError, PreprocessError, DetectionError, RenderError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
