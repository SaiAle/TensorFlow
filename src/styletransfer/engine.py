"""The style-transfer engine: load the model once, run inference.

This is the only module that touches the ML model. The pretrained Magenta
network is loaded lazily and cached in-process.
"""

import tensorflow as tf
import tensorflow_hub as hub

from .config import HUB_HANDLE, configure_cache

_model = None


def load_model(handle: str = HUB_HANDLE):
    """Load (and memoize) the pretrained stylization model."""
    global _model
    if _model is None:
        configure_cache()
        _model = hub.load(handle)
    return _model


def stylize(content_image: tf.Tensor, style_image: tf.Tensor, model=None) -> tf.Tensor:
    """Blend ``content_image`` with ``style_image``.

    Args:
        content_image: 4-D tensor ``[1, H, W, 3]`` in ``[0, 1]``.
        style_image: 4-D tensor ``[1, H, W, 3]`` in ``[0, 1]``.
        model: optional preloaded model; loaded on demand if omitted.

    Returns:
        Stylized image as a 4-D tensor ``[1, H, W, 3]`` in ``[0, 1]``.
    """
    model = model or load_model()
    outputs = model(tf.constant(content_image), tf.constant(style_image))
    return outputs[0]
