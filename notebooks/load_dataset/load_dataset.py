from tensorflow.keras.datasets import cifar10

# Завантажимо датасет
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Створимо список класів
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
