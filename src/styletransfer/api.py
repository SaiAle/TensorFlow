"""FastAPI service exposing style transfer over HTTP.

Run with:
    uvicorn styletransfer.api:app --reload
Then POST two image files to /stylize, or open /docs for the interactive UI.
"""

from contextlib import asynccontextmanager

import tensorflow as tf
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import Response

from .config import DEFAULT_CONTENT_SIZE, DEFAULT_STYLE_SIZE
from .engine import blend_strength, load_model, stylize
from .images import load_image_bytes


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the model cache once at startup so the first request isn't slow.
    load_model()
    yield


app = FastAPI(title="Fast Style Transfer", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/stylize")
async def stylize_endpoint(
    content: UploadFile = File(..., description="Content image."),
    style: UploadFile = File(..., description="Style image."),
    content_size: int = Query(DEFAULT_CONTENT_SIZE, ge=64, le=1024),
    style_size: int = Query(DEFAULT_STYLE_SIZE, ge=64, le=512),
    strength: float = Query(1.0, ge=0.0, le=1.0, description="Style strength."),
    style_blur: bool = Query(True, description="Smooth the style image."),
):
    """Stylize ``content`` with ``style`` and return a PNG image."""
    content_img = load_image_bytes(await content.read(), (content_size, content_size))
    style_img = load_image_bytes(await style.read(), (style_size, style_size))
    if style_blur:
        style_img = tf.nn.avg_pool(style_img, ksize=[3, 3], strides=[1, 1], padding="SAME")

    result = stylize(content_img, style_img)
    if strength < 1.0:
        result = blend_strength(content_img, result, strength)

    img = tf.squeeze(result, axis=0)
    img = tf.image.convert_image_dtype(tf.clip_by_value(img, 0.0, 1.0), tf.uint8)
    png = tf.image.encode_png(img).numpy()
    return Response(content=png, media_type="image/png")
