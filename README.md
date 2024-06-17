# CIFAR-10 Image Classification Application

This repository contains a comprehensive image classification application based on the CIFAR-10 dataset. The application includes three Jupyter notebooks for training different neural network architectures (MobileNetV2, InceptionV3, and ResNet50) and a web interface built with Gradio that allows users to upload images and obtain predictions from these trained models.

## Usage

- **_Uploading Images_**: Click on the "Upload Image" button to select an image from your computer.
- **_Selecting a Model_**: Use the dropdown menu to choose between the MobileNetV2, InceptionV3, and ResNet50 models.
- **_Getting Predictions_**: After uploading an image and selecting a model, the application will display the top predictions along with their confidence scores.

## Repository Structure

- **notebooks/**
  - **MobileNet_based.ipynb**: Jupyter notebook for training the MobileNetV2 model.
  - **Inception_based.ipynb**: Jupyter notebook for training the InceptionV3 model.
  - **resnet_model.ipynb**: Jupyter notebook for training the ResNet50 model.
  - **load_dataset/**
    - **load_dataset.py**: Script for loading and preprocessing the CIFAR-10 dataset.
- **models/**
  - **mn_model.keras**: Trained MobileNetV2 model.
  - **inception_v3.keras**: Trained InceptionV3 model.
  - **resnet_best.h5**: Trained ResNet50 model.
- **main.py**: Python script for launching the Gradio web interface.
- **results/**:
  - **functions.py**: Script for visualizing the results of the models
- **src/**
  - **examples**: Directory where examples of images are stored for testing neural network models
- **gitattributes**: Attribute files
- **gitignore**: Ignore files
- **requirements.txt**: A file with a list of packages for the correct operation of the program

## Setup Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/Balorum/ds_project.git
   cd ds_project
   ```

2. Create a virtual environment and activate it (for example python venv):

   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

## Training the Models

Navigate to the notebooks directory and open the respective notebooks for training:

- **Inception_based.ipynb**
- **MobileNet_based.ipynb**
- **resnet_model.ipynb**

Run the cells in each notebook to train the models. The trained models will be saved in the models directory.

## Running the Gradio Web Interface

Ensure the trained models are present in the models directory.

Run the Gradio app:

Install the required packages:

    ```
    python main.py
    ```

Open the link provided by Gradio in your web browser. You should see a web interface where you can upload images and get predictions from the trained models.

### _An example of the image classification application:_

![An example of the image classification application](https://res.cloudinary.com/dtjnagvw1/image/upload/v1718464282/ds_project/example_ipsko3.png)

## **_Models used_**

### MobileNetV2

_MobileNet V2 provides an efficient neural network architecture that is particularly well-suited for mobile and embedded vision applications. Its use of inverted residuals, linear bottlenecks, and depthwise separable convolutions allows it to achieve high performance with minimal computational resources._

_Accuracy obtained:_ ≈0.84

_Loss obtained:_ ≈0.98

#### Charts

<div align="center">_MobileNetV2 Accuracy_</div>
![Mobile_acc](https://res.cloudinary.com/dtjnagvw1/image/upload/v1718630103/ds_project/accuracy_mobilenet.png)

_MobileNetV2 Loss_
![Mobile_loss](https://res.cloudinary.com/dtjnagvw1/image/upload/v1718630265/ds_project/loss_mobilenet.png)

### [**Link demo**](https://huggingface.co/spaces/the10or/class_pic)
