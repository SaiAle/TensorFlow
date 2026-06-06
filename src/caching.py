from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def cache_path(cache_dir: Path, content_key: str) -> Path:
    import hashlib

    digest = hashlib.sha256(content_key.encode()).hexdigest()[:16]
    return cache_dir / f"{digest}.json"


def load_cache(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def save_cache(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
