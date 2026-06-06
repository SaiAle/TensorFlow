from __future__ import annotations

from pathlib import Path


def ensure(path: Path) -> Path:
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)
