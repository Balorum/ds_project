from tensorflow.keras.datasets import cifar10
import numpy as np
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Define class names
classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

x_test_base = np.array(x_test, dtype=np.float16)
x_test_base = x_test_base / 255
y_test_base = to_categorical(y_test, 10)

x_test_mob = x_test
y_test_mob = y_test
