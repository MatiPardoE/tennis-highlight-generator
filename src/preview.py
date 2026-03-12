from __future__ import annotations

from src.models import TimeSegment


def format_seconds(value: float) -> str:
    total = int(round(value))
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def segments_to_rows(segments: list[TimeSegment]) -> list[dict[str, str | float | int]]:
    rows: list[dict[str, str | float | int]] = []
    for idx, segment in enumerate(segments, start=1):
        rows.append(
            {
                "#": idx,
                "Tipo": segment.label,
                "Inicio (s)": round(segment.start_sec, 2),
                "Fin (s)": round(segment.end_sec, 2),
                "Duración (s)": round(segment.duration_sec, 2),
                "Inicio": format_seconds(segment.start_sec),
                "Fin": format_seconds(segment.end_sec),
            }
        )
    return rows


def total_duration(segments: list[TimeSegment]) -> float:
    return sum(segment.duration_sec for segment in segments)
