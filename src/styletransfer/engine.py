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


def blend_strength(content_image: tf.Tensor, stylized_image: tf.Tensor, strength: float) -> tf.Tensor:
    """Interpolate between the content and stylized image to control style strength.

    ``strength`` of 1.0 returns the fully stylized image; 0.0 returns the
    original content. The content image is resized to the stylized image's
    resolution before blending.
    """
    if strength >= 1.0:
        return stylized_image
    target = tf.shape(stylized_image)[1:3]
    content_resized = tf.image.resize(content_image, target)
    return strength * stylized_image + (1.0 - strength) * content_resized
