from __future__ import annotations

from pathlib import Path

import pytest

from src.core import tensor_to_image
from src.models import resolve


def test_default_model_is_string() -> None:
    assert isinstance(resolve(), str)
    assert "tfhub.dev" in resolve()


def test_alias_resolution() -> None:
    resolved = resolve(alias="magenta")
    assert resolved


def test_tensor_to_image_returns_image() -> None:
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        pytest.skip("numpy/PIL unavailable")

    arr = np.zeros((64, 64, 3), dtype="float32")
    img = tensor_to_image(arr)
    assert isinstance(img, Image.Image)
    assert img.size == (64, 64)
