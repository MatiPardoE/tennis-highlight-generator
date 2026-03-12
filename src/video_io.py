from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2

from src.models import VideoMetadata


class VideoIOError(RuntimeError):
    pass


def save_uploaded_video(data: bytes, original_name: str, workspace: Path) -> Path:
    workspace.mkdir(parents=True, exist_ok=True)
    ext = Path(original_name).suffix.lower() or ".mp4"
    input_path = workspace / f"input{ext}"
    input_path.write_bytes(data)
    return input_path


def read_video_metadata(video_path: Path) -> VideoMetadata:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoIOError(f"No se pudo abrir el video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0.0:
        raise VideoIOError("FPS inválido. El archivo puede estar corrupto.")

    duration_sec = frame_count / fps if frame_count > 0 else 0.0
    return VideoMetadata(
        path=video_path,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_sec=duration_sec,
    )


def ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise VideoIOError("No se encontró ffmpeg en PATH.")
    if shutil.which("ffprobe") is None:
        raise VideoIOError("No se encontró ffprobe en PATH.")


def has_audio_stream(video_path: Path) -> bool:
    cmd = [
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
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return False
    return "audio" in proc.stdout
