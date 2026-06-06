from __future__ import annotations
import os, tempfile
from pathlib import Path
from PIL import Image
import sys
sys.path.insert(0, os.getcwd())

from src.models import resolve
from src.caching import cache_path, load_cache, save_cache
from src.inference import stylize_bytes

# probe availability without relying on TF-Hub import
TFHUB_AVAILABLE = False
try:
    import tensorflow_hub as hub
    TFHUB_AVAILABLE = True
except Exception as e:
    print("tensorflow-hub probe failed:", repr(e))

model_url = resolve()
print("default model:", model_url)

with tempfile.TemporaryDirectory() as d:
    base = Path(d)
    content = base / 'content.png'
    style = base / 'style.png'
    out = base / 'out.png'
    Image.new('RGB', (32, 32), (40, 80, 120)).save(content)
    Image.new('RGB', (32, 32), (200, 160, 60)).save(style)

    print("tensorflow-hub available:", TFHUB_AVAILABLE)
    if not TFHUB_AVAILABLE:
        raise SystemExit("Smoke test stopped: tensorflow-hub could not be imported in this environment.")

    png = stylize_bytes(content.read_bytes(), style.read_bytes(), model_url)
    out.write_bytes(png)
    print("wrote output:", out, out.stat().st_size, "bytes")
