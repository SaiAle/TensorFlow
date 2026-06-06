from __future__ import annotations

from pathlib import Path

import pytest

from src.models import resolve
from src.worker import run_batch


def _write_image(path: Path, color: tuple[int, int, int] = (0, 0, 0)) -> None:
    try:
        from PIL import Image
    except Exception:
        pytest.skip("PIL not installed")
    Image.new("RGB", (16, 16), color).save(path)


def test_run_batch_writes_output(tmp_path: Path) -> None:
    content_dir = tmp_path / "content"
    style_dir = tmp_path / "style"
    out_dir = tmp_path / "output"
    content_dir.mkdir()
    style_dir.mkdir()
    _write_image(content_dir / "c1.png")
    _write_image(style_dir / "s1.png", color=(128, 128, 128))

    records = run_batch(
        content_dir,
        style_dir,
        out_dir,
        model_url=resolve(),
        cache_dir=tmp_path / "cache",
    )
    assert (out_dir / "c1__s1.png").exists()
    assert (out_dir / "manifest.json").exists()
    assert records and "out" in records[0]
