import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("models/newmodel.h5")


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

print(model.summary())
iface.launch()
