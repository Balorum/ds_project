import gradio as gr
import numpy as np
import tensorflow as tf


def process_image(image):

    img_array = tf.image.resize(image, (32, 32))

    # model working...

    return img_array


iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(label="Upload an Image"),
    outputs=gr.Textbox(label="32x32 Image Tensor"),
    title="Image recognition App\n",
    examples=["src/examples/cat.jpg", "src/examples/deer.jpg", "src/examples/frog.jpg"],
)


iface.launch()
