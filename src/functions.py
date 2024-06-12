import numpy as np
import matplotlib.pyplot as plt
from .load_dataset.load_dataset import x_test, y_test, x_train, y_train, classes


def visualize_predictions(model, n=15, rows=3):
    start = np.random.randint(0, len(x_test) - n)
    end = start + n
    to_predict = x_test[start:end]
    predicted = model.predict(to_predict)
    real = y_test[start:end]

    cols = (n + rows - 1) // rows

    print(f" Labels:      {real.T[0]}")
    print(f" Predictions: {np.argmax(predicted, axis=1)}")

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))

    for i in range(n):
        row = i // cols
        col = i % cols
        axes[row, col].imshow(to_predict[i])
        axes[row, col].set_title(f"predicted: {classes[np.argmax(predicted[i])]}\nreal label: {classes[real[i][0]]}")
        axes[row, col].axis('off')

    for i in range(n, rows * cols):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    plt.show()

def history_plot(history):
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.legend(["accuracy", "val_accuracy"])
    plt.grid()
    plt.title("Accuracy")
    plt.show()
    plt.legend(["loss", "val_loss"])
    plt.title("Loss")
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.grid()
    plt.show()