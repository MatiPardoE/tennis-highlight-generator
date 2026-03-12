from __future__ import annotations

import re
import tempfile
from pathlib import Path


def create_workspace() -> Path:
    return Path(tempfile.mkdtemp(prefix="tennis_highlights_"))


def sanitize_stem(filename: str) -> str:
    stem = Path(filename).stem or "video"
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem).strip("_")
    return clean or "video"
