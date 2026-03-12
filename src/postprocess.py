from __future__ import annotations

from src.config import ProcessingConfig
from src.models import TimeSegment


def _sort_segments(segments: list[TimeSegment]) -> list[TimeSegment]:
    return sorted(segments, key=lambda s: s.start_sec)


def merge_close_segments(segments: list[TimeSegment], max_gap_sec: float) -> list[TimeSegment]:
    if not segments:
        return []

    ordered = _sort_segments(segments)
    merged: list[TimeSegment] = [ordered[0]]

    for current in ordered[1:]:
        last = merged[-1]
        gap = current.start_sec - last.end_sec
        if gap <= max_gap_sec:
            last.end_sec = max(last.end_sec, current.end_sec)
        else:
            merged.append(TimeSegment(current.start_sec, current.end_sec, current.label))

    return merged


def remove_short_segments(segments: list[TimeSegment], min_duration_sec: float) -> list[TimeSegment]:
    return [segment for segment in segments if segment.duration_sec >= min_duration_sec]


def apply_padding(segments: list[TimeSegment], padding_sec: float, video_duration_sec: float) -> list[TimeSegment]:
    padded: list[TimeSegment] = []
    for segment in segments:
        start = max(0.0, segment.start_sec - padding_sec)
        end = min(video_duration_sec, segment.end_sec + padding_sec)
        if end > start:
            padded.append(TimeSegment(start_sec=start, end_sec=end, label=segment.label))
    return padded


def build_non_game_segments(game_segments: list[TimeSegment], video_duration_sec: float) -> list[TimeSegment]:
    if not game_segments:
        return [TimeSegment(0.0, video_duration_sec, label="NON_GAME")]

    non_game: list[TimeSegment] = []
    cursor = 0.0

    for segment in _sort_segments(game_segments):
        if segment.start_sec > cursor:
            non_game.append(TimeSegment(cursor, segment.start_sec, label="NON_GAME"))
        cursor = max(cursor, segment.end_sec)

    if cursor < video_duration_sec:
        non_game.append(TimeSegment(cursor, video_duration_sec, label="NON_GAME"))

    return non_game


def postprocess_game_segments(
    raw_segments: list[TimeSegment],
    cfg: ProcessingConfig,
    video_duration_sec: float,
) -> list[TimeSegment]:
    merged = merge_close_segments(raw_segments, cfg.min_pause_sec)
    filtered = remove_short_segments(merged, cfg.min_game_sec)
    padded = apply_padding(filtered, cfg.padding_sec, video_duration_sec)
    return merge_close_segments(padded, max_gap_sec=0.01)
