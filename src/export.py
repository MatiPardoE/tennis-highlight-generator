from __future__ import annotations

import subprocess
from pathlib import Path

from src.config import ExportConfig
from src.models import TimeSegment
from src.video_io import has_audio_stream


class ExportError(RuntimeError):
    pass


def _fmt_time(value: float) -> str:
    return f"{max(0.0, value):.3f}"


def _build_filter_complex(segments: list[TimeSegment], include_audio: bool) -> str:
    parts: list[str] = []

    for idx, segment in enumerate(segments):
        start = _fmt_time(segment.start_sec)
        end = _fmt_time(segment.end_sec)
        parts.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{idx}]")
        if include_audio:
            parts.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{idx}]")

    if include_audio:
        concat_inputs = "".join(f"[v{idx}][a{idx}]" for idx in range(len(segments)))
        parts.append(f"{concat_inputs}concat=n={len(segments)}:v=1:a=1[vout][aout]")
    else:
        concat_inputs = "".join(f"[v{idx}]" for idx in range(len(segments)))
        parts.append(f"{concat_inputs}concat=n={len(segments)}:v=1:a=0[vout]")

    return ";".join(parts)


def export_highlight_video(
    input_path: Path,
    output_path: Path,
    segments: list[TimeSegment],
    cfg: ExportConfig,
) -> Path:
    if not segments:
        raise ExportError("No hay segmentos GAME para exportar.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    include_audio = has_audio_stream(input_path)
    filter_complex = _build_filter_complex(segments, include_audio)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-c:v",
        "libx264",
        "-preset",
        cfg.preset,
        "-crf",
        str(cfg.crf),
    ]

    if include_audio:
        cmd.extend(["-map", "[aout]", "-c:a", "aac", "-b:a", "128k"])
    else:
        cmd.append("-an")

    cmd.append(str(output_path))

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        stderr_tail = "\n".join(proc.stderr.splitlines()[-20:])
        raise ExportError(f"ffmpeg falló al exportar:\n{stderr_tail}")

    return output_path
