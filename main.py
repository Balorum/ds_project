import tensorflow as tf
import gradio as gr
import os

from notebooks.load_dataset.load_dataset import classes

nm_model = tf.keras.models.load_model("models/mn_model.keras")

resnet_model = tf.keras.models.load_model("models/resnet_best.h5")

inception_model = tf.keras.models.load_model("models/inception_v3.keras")

cifar10_labels = classes
models = ["ResNetBased Model", "MobileNetBased Model", "InceptionBased Model"]


def classify_image(input_image, model_name):
    try:
        input_image = tf.image.resize(input_image, (32, 32))
        labels = cifar10_labels
        model = get_model(model_name)
        input_image = tf.expand_dims(input_image, axis=0)
        predictions = model.predict(input_image).flatten()
        top_indices = predictions.argsort()[-10:][::-1]
        confidences = {labels[i]: float(predictions[i]) for i in top_indices}
        return confidences
    except Exception as e:
        return {"error": str(e)}


def get_model(model_name):
    if model_name == "MobileNetBased Model":
        return nm_model
    elif model_name == "ResNetBased Model":
        return resnet_model
    elif model_name == "InceptionBased Model":
        return inception_model


interface = gr.Interface(
    fn=classify_image,
    inputs=[
        gr.Image(type="numpy", image_mode="RGB", label="Input Image"),
        gr.Dropdown(models, label="Model Choice"),
    ],
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
)

interface.launch(debug=False, share=True)
