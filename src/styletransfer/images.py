"""Image loading, preprocessing, and saving.

Extracted from the original Colab notebook and generalized so it works with
local file paths *and* URLs (the notebook was locked to ``google.colab.files``).
All I/O uses TensorFlow natively, so matplotlib is not required at runtime.
"""

import os

import tensorflow as tf


def crop_center(image: tf.Tensor) -> tf.Tensor:
    """Return the largest centered square crop of a 4-D image tensor.

    Args:
        image: Tensor of shape ``[1, H, W, C]``.
    """
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    return tf.image.crop_to_bounding_box(image, offset_y, offset_x, new_shape, new_shape)


def _resolve(source: str) -> str:
    """Return a local file path for ``source``, downloading it if it is a URL."""
    if source.startswith(("http://", "https://")):
        # Keep the cached filename short and unique-ish, mirroring the notebook.
        return tf.keras.utils.get_file(os.path.basename(source)[-128:], source)
    return source


def load_image(source: str, image_size=(256, 256)) -> tf.Tensor:
    """Load an image from a path or URL into a normalized 4-D tensor.

    Decoding forces 3 channels, which transparently handles grayscale and
    RGBA inputs. The result is float32 in ``[0, 1]`` with shape ``[1, H, W, 3]``.
    """
    path = _resolve(source)
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # -> [0, 1]
    img = img[tf.newaxis, ...]
    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img


def preprocess_array(array, image_size=(256, 256)) -> tf.Tensor:
    """Preprocess an in-memory image (e.g. from a web UI) like :func:`load_image`.

    Accepts an HxW, HxWx1, HxWx3, or HxWx4 numpy array / tensor and returns a
    normalized 4-D tensor ``[1, H, W, 3]`` in ``[0, 1]``.
    """
    img = tf.convert_to_tensor(array)
    img = tf.image.convert_image_dtype(img, tf.float32)  # -> [0, 1]
    if img.shape.rank == 2:
        img = tf.expand_dims(img, -1)
    if img.shape[-1] == 4:
        img = img[..., :3]  # drop alpha
    if img.shape[-1] == 1:
        img = tf.image.grayscale_to_rgb(img)
    img = img[tf.newaxis, ...]
    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img


def to_numpy(image: tf.Tensor):
    """Convert a 4-D float image tensor (``[1, H, W, 3]``) to a uint8 numpy array."""
    img = tf.squeeze(image, axis=0)
    img = tf.image.convert_image_dtype(tf.clip_by_value(img, 0.0, 1.0), tf.uint8)
    return img.numpy()


def save_image(image: tf.Tensor, path: str) -> None:
    """Write a 4-D float image tensor (``[1, H, W, 3]``, range ``[0, 1]``) to disk.

    Format is inferred from the file extension (``.jpg``/``.jpeg`` -> JPEG,
    everything else -> PNG).
    """
    img = tf.squeeze(image, axis=0)
    img = tf.image.convert_image_dtype(tf.clip_by_value(img, 0.0, 1.0), tf.uint8)
    ext = os.path.splitext(path)[1].lower()
    data = tf.image.encode_jpeg(img) if ext in (".jpg", ".jpeg") else tf.image.encode_png(img)
    tf.io.write_file(path, data)
