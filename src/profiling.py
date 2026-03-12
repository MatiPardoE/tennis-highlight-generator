from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterator


@dataclass(slots=True)
class StageProfiler:
    stage_seconds: dict[str, float] = field(default_factory=dict)
    _started_at: float = field(default_factory=perf_counter)

    @contextmanager
    def track(self, stage_name: str) -> Iterator[None]:
        started_at = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - started_at
            self.stage_seconds[stage_name] = self.stage_seconds.get(stage_name, 0.0) + elapsed

    @property
    def total_seconds(self) -> float:
        return perf_counter() - self._started_at

