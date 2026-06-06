"""Integration tests for the stylization engine.

These download the pretrained TF-Hub model on first run, so they are marked
``integration`` and excluded from the default/offline CI run. Run them with:

    pytest -m integration
"""

import numpy as np
import pytest
import tensorflow as tf

from styletransfer.engine import blend_strength, stylize


def test_blend_strength_endpoints():
    """blend_strength is offline-safe: it does not touch the model."""
    content = tf.zeros([1, 8, 8, 3])
    stylized = tf.ones([1, 8, 8, 3])
    # strength=1 -> fully stylized; strength=0 -> fully content.
    assert float(tf.reduce_mean(blend_strength(content, stylized, 1.0))) == 1.0
    assert float(tf.reduce_mean(blend_strength(content, stylized, 0.0))) == 0.0
    mid = blend_strength(content, stylized, 0.5)
    assert abs(float(tf.reduce_mean(mid)) - 0.5) < 1e-6


@pytest.mark.integration
def test_stylize_output_contract():
    """End-to-end: output is a 4-D RGB image in [0, 1] matching the content size."""
    content = tf.random.uniform([1, 256, 256, 3])
    style = tf.random.uniform([1, 256, 256, 3])
    out = stylize(content, style)
    assert out.shape.rank == 4 and out.shape[-1] == 3
    arr = out.numpy()
    assert arr.min() >= 0.0 and arr.max() <= 1.0
    assert not np.allclose(arr, content.numpy())  # actually transformed
