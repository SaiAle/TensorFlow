"""Project-wide configuration and defaults."""

import os
from pathlib import Path

# Pretrained arbitrary image stylization model (Magenta, via TF-Hub).
HUB_HANDLE = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"

# The style-prediction network was trained at 256x256; that is the recommended
# style image size. Content size is arbitrary.
DEFAULT_CONTENT_SIZE = 384
DEFAULT_STYLE_SIZE = 256

# Cache the (re)downloaded TF-Hub model and example images under the project so
# the first run pays the download cost once and later runs are fast/offline.
CACHE_DIR = Path(os.environ.get("STYLETRANSFER_CACHE", Path.home() / ".cache" / "styletransfer"))


def configure_cache() -> None:
    """Point TF-Hub at our local cache dir unless the user already set one."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TFHUB_CACHE_DIR", str(CACHE_DIR / "tfhub"))
