from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from config import PipelineManifest, RenderConfig, Segment
from preprocess import build_geometry_filter_chain, read_video_metadata


class RenderError(RuntimeError):
    pass


def ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise RenderError("No se encontró ffmpeg en PATH.")


def _has_audio_stream(video_path: Path) -> bool:
    if shutil.which("ffprobe") is None:
        return False

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video_path),
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    return proc.returncode == 0 and "audio" in proc.stdout


def _fmt_time(value: float) -> str:
    return f"{max(0.0, value):.3f}"


def _build_filter_complex(
    segments: list[Segment],
    include_audio: bool,
    final_video_filters: str | None = None,
) -> str:
    parts: list[str] = []

    for idx, segment in enumerate(segments):
        start = _fmt_time(segment.start_sec)
        end = _fmt_time(segment.end_sec)
        parts.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{idx}]")
        if include_audio:
            parts.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{idx}]")

    concat_video_label = "vcat"
    if include_audio:
        concat_inputs = "".join(f"[v{idx}][a{idx}]" for idx in range(len(segments)))
        parts.append(f"{concat_inputs}concat=n={len(segments)}:v=1:a=1[{concat_video_label}][aout]")
    else:
        concat_inputs = "".join(f"[v{idx}]" for idx in range(len(segments)))
        parts.append(f"{concat_inputs}concat=n={len(segments)}:v=1:a=0[{concat_video_label}]")

    if final_video_filters:
        parts.append(f"[{concat_video_label}]{final_video_filters}[vout]")
    else:
        parts.append(f"[{concat_video_label}]null[vout]")

    return ";".join(parts)


def render_highlights(
    manifest_path: Path,
    output_path: Path,
    render_cfg: RenderConfig,
) -> Path:
    ensure_ffmpeg_available()
    manifest = PipelineManifest.load(manifest_path.expanduser().resolve())

    if not manifest.useful_segments:
        raise RenderError("No hay segmentos útiles para exportar.")

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    include_audio = _has_audio_stream(manifest.original_path)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(manifest.original_path),
    ]
    final_video_filters: str | None = None
    if render_cfg.apply_calibrated_geometry:
        metadata = read_video_metadata(manifest.original_path)
        geometry_filters, _, _ = build_geometry_filter_chain(
            source_width=metadata.width,
            source_height=metadata.height,
            rotation=manifest.rotation,
            roi=manifest.roi if render_cfg.apply_roi_crop else None,
        )
        final_parts: list[str] = []
        if geometry_filters:
            final_parts.append(geometry_filters)
        final_video_filters = ",".join(final_parts) or None

    filter_complex = _build_filter_complex(
        manifest.useful_segments,
        include_audio=include_audio,
        final_video_filters=final_video_filters,
    )
    video_map = "[vout]"

    command.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            video_map,
            "-c:v",
            render_cfg.video_codec,
            "-preset",
            render_cfg.preset,
            "-crf",
            str(render_cfg.crf),
            "-movflags",
            "+faststart",
        ]
    )

    if include_audio:
        command.extend(
            [
                "-map",
                "[aout]",
                "-c:a",
                render_cfg.audio_codec,
                "-b:a",
                render_cfg.audio_bitrate,
            ]
        )
    else:
        command.append("-an")

    command.append(str(output_path))

    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode == 0:
        return output_path

    stderr_tail = "\n".join(proc.stderr.splitlines()[-20:])
    raise RenderError(f"ffmpeg falló al exportar highlights:\n{stderr_tail}")
