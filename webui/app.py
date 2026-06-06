"""Gradio web UI for fast style transfer.

Run with:  python webui/app.py
Then open the printed local URL in a browser.
"""

import gradio as gr
import tensorflow as tf

from styletransfer import preprocess_array, stylize, to_numpy
from styletransfer.config import DEFAULT_CONTENT_SIZE, DEFAULT_STYLE_SIZE
from styletransfer.engine import blend_strength, load_model


def run(content_img, style_img, content_size, style_size, blur_style, strength):
    if content_img is None or style_img is None:
        raise gr.Error("Please provide both a content image and a style image.")
    content = preprocess_array(content_img, (int(content_size), int(content_size)))
    style = preprocess_array(style_img, (int(style_size), int(style_size)))
    if blur_style:
        style = tf.nn.avg_pool(style, ksize=[3, 3], strides=[1, 1], padding="SAME")
    result = stylize(content, style)
    if strength < 1.0:
        result = blend_strength(content, result, float(strength))
    return to_numpy(result)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Fast Style Transfer") as demo:
        gr.Markdown("# Fast Style Transfer\nBlend a content image with a style image.")
        with gr.Row():
            content_in = gr.Image(label="Content", type="numpy")
            style_in = gr.Image(label="Style", type="numpy")
        with gr.Row():
            content_size = gr.Slider(128, 768, value=DEFAULT_CONTENT_SIZE, step=32, label="Content size")
            style_size = gr.Slider(128, 512, value=DEFAULT_STYLE_SIZE, step=32, label="Style size")
            blur_style = gr.Checkbox(value=True, label="Smooth style image")
        strength = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Style strength")
        go = gr.Button("Stylize", variant="primary")
        output = gr.Image(label="Stylized")
        go.click(
            run,
            [content_in, style_in, content_size, style_size, blur_style, strength],
            output,
        )
    return demo


if __name__ == "__main__":
    load_model()  # warm the model cache before serving
    build_demo().launch()
