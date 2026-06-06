"""Unit tests for image I/O — no network or model required."""

import numpy as np
import tensorflow as tf

from styletransfer.images import crop_center, load_image, save_image


def _write_sample(path, h, w, channels=3):
    arr = (np.random.rand(h, w, channels) * 255).astype(np.uint8)
    data = tf.image.encode_png(tf.constant(arr))
    tf.io.write_file(str(path), data)


def test_crop_center_makes_square():
    img = tf.zeros([1, 100, 200, 3])
    cropped = crop_center(img)
    assert cropped.shape[1] == cropped.shape[2] == 100


def test_load_image_shape_and_range(tmp_path):
    src = tmp_path / "in.png"
    _write_sample(src, 120, 200)
    img = load_image(str(src), (64, 64))
    assert img.shape[0] == 1 and img.shape[-1] == 3
    assert float(tf.reduce_min(img)) >= 0.0
    assert float(tf.reduce_max(img)) <= 1.0


def test_load_image_handles_grayscale(tmp_path):
    src = tmp_path / "gray.png"
    _write_sample(src, 80, 80, channels=1)
    img = load_image(str(src), (32, 32))
    assert img.shape[-1] == 3  # decoded to 3 channels


def test_save_round_trip(tmp_path):
    img = tf.random.uniform([1, 50, 50, 3])
    out = tmp_path / "out.png"
    save_image(img, str(out))
    assert out.exists() and out.stat().st_size > 0
