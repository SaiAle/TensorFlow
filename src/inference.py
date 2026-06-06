from __future__ import annotations

import importlib
import io
import sys
import types
from pathlib import Path
from typing import Union

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

from PIL import Image

try:
    import tensorflow as tf
except Exception as _tf_exc:  # pragma: no cover - environment guard
    raise RuntimeError(f"tensorflow is required: {_tf_exc}") from _tf_exc

# tensorflow_hub import-time dependency workaround in this environment:
# `_ensure_tf_install()` imports `pkg_resources.parse_version`. If `setuptools`
# isn't importable in the same process, shim it instead of failing hard.
if "pkg_resources" not in sys.modules:
    pkg_shim = types.ModuleType("pkg_resources")

    def _parse_version(_v: str):  # pragma: no cover - minimal stub
        return _v

    pkg_shim.parse_version = _parse_version  # type: ignore[attr-defined]
    sys.modules["pkg_resources"] = pkg_shim

try:
    import tensorflow_hub as hub
except ModuleNotFoundError:  # pragma: no cover - handled below
    hub = None  # type: ignore[assignment]
except Exception as _hub_exc:  # pragma: no cover - environment guard
    raise RuntimeError(f"tensorflow-hub is required: {_hub_exc}") from _hub_exc

from .models import resolve


def _decode(content: bytes) -> "np.ndarray":
    img = Image.open(io.BytesIO(content)).convert("RGB")
    return np.array(img)


def _encode_png(array: "np.ndarray") -> bytes:
    img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def stylize_bytes(
    content: bytes,
    style: bytes,
    model_url: str,
    *,
    preserve_aspect_ratio: bool = True,
) -> bytes:
    if hub is None:
        raise RuntimeError("tensorflow-hub is not installed in this environment")

    content_image = _decode(content)
    style_image = _decode(style)

    hub_module = hub.load(model_url)

    content_tensor = tf.constant(content_image[np.newaxis, ...], dtype=tf.float32)
    style_tensor = tf.constant(style_image[np.newaxis, ...], dtype=tf.float32)

    outputs = hub_module(content_tensor, style_tensor)
    result = outputs[0][0].numpy()

    return _encode_png(result)
