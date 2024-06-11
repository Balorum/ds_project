import sys
import os

sys.path.insert(0, os.getcwd())

import tensorflow as tf
import gradio as gr
import requests

from notebooks.load_dataset.dataset import classes
import numpy as np

nm_model = tf.keras.models.load_model("models/mn_model.keras")

resnet_model = tf.keras.models.load_model("models/resnet_best.h5")

cifar10_labels = classes


def classify_image(inp, model_choice):
    try:
        print("Original image shape:", inp.shape)

        if model_choice == "MobileNetBased Model":
            inp = tf.image.resize(inp, (32, 32))

            print("Processed image for MobileNet model:", inp)
            model = nm_model
            labels = cifar10_labels
        elif model_choice == "ResNetBased Model":
            inp = tf.image.resize(inp, (32, 32))
            inp = tf.keras.applications.resnet.preprocess_input(inp)
            print("Processed image for ResNet model:", inp)
            model = resnet_model
            labels = cifar10_labels

        print("Resized image shape:", inp.shape)

        inp = tf.expand_dims(inp, axis=0)
        print("Input to the model:", inp)

        prediction = model.predict(inp).flatten()
        print("Predictions:", prediction)

        if model_choice == "MobileNetV2":
            top_indices = prediction.argsort()[-10:][::-1]
            confidences = {labels[i]: float(prediction[i]) for i in top_indices}
        else:
            confidences = {labels[i]: float(prediction[i]) for i in range(len(labels))}

        return confidences
    except Exception as e:
        return {"error": str(e)}


interface = gr.Interface(
    fn=classify_image,
    inputs=[
        gr.Image(type="numpy", image_mode="RGB", label="Input Image"),
        gr.Dropdown(
            ["ResNetBased Model", "MobileNetBased Model"],
            value="ResNetBased Model",
            label="Model Choice",
        ),
    ],
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
)

interface.launch(debug=False, share=True)
