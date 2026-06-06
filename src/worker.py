from __future__ import annotations

import json
from pathlib import Path

from .core import apply_style
from .models import resolve
from .utils import ensure

try:
    import numpy as np
    from PIL import Image
except Exception:
    np = None
    Image = None


def run_batch(
    content_dir: Path,
    style_dir: Path,
    out_dir: Path,
    model_url: str,
    *,
    cache_dir: Path | None = None,
    pattern: str = "*.png,*.jpg,*.jpeg",
) -> list[dict]:
    ensure(content_dir)
    ensure(style_dir)
    ensure(out_dir)

    content_paths = _resolve(content_dir, pattern)
    style_paths = _resolve(style_dir, pattern)

    if not content_paths:
        raise ValueError(f"no content files under {content_dir}")
    if not style_paths:
        raise ValueError(f"no style files under {style_dir}")

    if cache_dir is not None:
        ensure(cache_dir)

    records: list[dict] = []

    # exhaustively pair each content with each style
    for cp in content_paths:
        for sp in style_paths:
            out = out_dir / f"{cp.stem}__{sp.stem}.png"
            summary = apply_style(
                cp,
                sp,
                out,
                model_url=model_url,
                cache_dir=cache_dir,
            )
            records.append(summary)

    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps(records, indent=2))
    return records


def _resolve(base: Path, pattern: str):
    matches: list[Path] = []
    for leaf in pattern.split(","):
        matches.extend(sorted(base.rglob(leaf.strip())))
    return matches
