"""Command-line interface for arbitrary fast style transfer.

Example:
    styletransfer content.jpg style.jpg -o stylized.png
    styletransfer https://.../photo.jpg https://.../wave.jpg -o out.png
"""

import argparse
import os
import sys

import tensorflow as tf

from .config import DEFAULT_CONTENT_SIZE, DEFAULT_STYLE_SIZE
from .engine import blend_strength, load_model, stylize
from .images import load_image, save_image

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="styletransfer",
        description="Blend a content image with a style image using a pretrained "
        "arbitrary neural style-transfer model.",
    )
    parser.add_argument("content", help="Content image (local path or URL).")
    parser.add_argument(
        "style", help="Style image (local path or URL), or a directory of styles for batch mode."
    )
    parser.add_argument(
        "-o", "--output", default="stylized.png",
        help="Output image path, or output directory when style is a directory "
        "(default: stylized.png).",
    )
    parser.add_argument(
        "--content-size", type=int, default=DEFAULT_CONTENT_SIZE,
        help=f"Content image size (default: {DEFAULT_CONTENT_SIZE}).",
    )
    parser.add_argument(
        "--style-size", type=int, default=DEFAULT_STYLE_SIZE,
        help=f"Style image size; 256 is recommended (default: {DEFAULT_STYLE_SIZE}).",
    )
    parser.add_argument(
        "--no-style-blur", action="store_true",
        help="Skip the average-pool smoothing applied to the style image.",
    )
    parser.add_argument(
        "--strength", type=float, default=1.0,
        help="Style strength in [0, 1]; 1.0 is fully stylized (default: 1.0).",
    )
    return parser


def _run_one(content, style_source, output, args, model) -> None:
    style = load_image(style_source, (args.style_size, args.style_size))
    if not args.no_style_blur:
        # Smoothing the style image (as in the original notebook) tends to give
        # cleaner results.
        style = tf.nn.avg_pool(style, ksize=[3, 3], strides=[1, 1], padding="SAME")
    stylized = stylize(content, style, model=model)
    if args.strength < 1.0:
        stylized = blend_strength(content, stylized, args.strength)
    save_image(stylized, output)
    print(f"Saved -> {output}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    print(f"Loading content: {args.content}")
    content = load_image(args.content, (args.content_size, args.content_size))
    print("Loading model...")
    model = load_model()

    if os.path.isdir(args.style):
        # Batch mode: stylize the content image with every style in the folder.
        styles = sorted(
            f for f in os.listdir(args.style) if f.lower().endswith(_IMAGE_EXTS)
        )
        if not styles:
            print(f"No images found in style directory: {args.style}", file=sys.stderr)
            return 1
        os.makedirs(args.output, exist_ok=True)
        print(f"Batch mode: {len(styles)} style(s) -> {args.output}")
        for name in styles:
            out = os.path.join(args.output, f"stylized_{os.path.splitext(name)[0]}.png")
            _run_one(content, os.path.join(args.style, name), out, args, model)
    else:
        print(f"Loading style:   {args.style}")
        print("Stylizing...")
        _run_one(content, args.style, args.output, args, model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
