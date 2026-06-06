from __future__ import annotations

from pathlib import Path

from .caching import cache_path, load_cache, save_cache
from .inference import stylize_bytes
from .models import resolve
from .utils import ensure

try:
    import numpy as np
    from PIL import Image
except Exception:  # non-image test environments
    np = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]


def apply_style(
    content_path: Path,
    style_path: Path,
    out_path: Path,
    model_url: str,
    *,
    preserve_aspect_ratio: bool = True,
    cache_dir: Path | None = None,
) -> dict:
    if np is None or Image is None:
        raise RuntimeError("numpy and Pillow are required")

    content_bytes = content_path.read_bytes()
    style_bytes = style_path.read_bytes()

    cache_material = (
        f"{model_url}|{content_path}:{len(content_bytes)}|{style_path}:{len(style_bytes)}"
    )
    cached = None
    if cache_dir is not None:
        cached = load_cache(cache_path(cache_dir, cache_material))
    if cached is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(cached.get("png", b""))
        return _summary(out_path, used_cache=True)

    png_bytes = stylize_bytes(content_bytes, style_bytes, model_url)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(png_bytes)

    if cache_dir is not None:
        save_cache(cache_path(cache_dir, cache_material), {"png": png_bytes})

    return _summary(out_path, used_cache=False)


def restore_from_cache(
    content_path: Path,
    style_path: Path,
    out_path: Path,
    model_url: str,
    cache_dir: Path,
) -> bool:
    cache_material = (
        f"{model_url}|{content_path}:{len(content_path.read_bytes())}|{style_path}:{len(style_path.read_bytes())}"
    )
    cached = load_cache(cache_path(cache_dir, cache_material))
    if cached is None:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(cached.get("png", b""))
    return True


def tensor_to_image(tensor) -> "Image.Image":
    try:
        array = tensor.numpy()
    except AttributeError:
        array = np.asarray(tensor)

    if np.max(array) <= 1:
        array = array * 255
    array = np.clip(array, 0, 255).astype("uint8")
    return Image.fromarray(array)


def save_tensor(path: Path, tensor) -> None:
    tensor_to_image(tensor).save(path)


def _summary(path: Path, *, used_cache: bool) -> dict:
    return {
        "out": str(path),
        "bytes": path.stat().st_size,
        "cached": used_cache,
    }
