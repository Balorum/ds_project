import numpy as np
import matplotlib.pyplot as plt
import os

print(os.getcwd())
from notebooks.load_dataset.load_dataset import (
    x_test,
    y_test,
    x_train,
    y_train,
    classes,
)


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
        axes[row, col].set_title(
            f"predicted: {classes[np.argmax(predicted[i])]}\nreal label: {classes[real[i][0]]}"
        )
        axes[row, col].axis("off")

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


def calculate_percent_right_mob(test, labels, model):
    y_pred_probs = model.predict(test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = labels

    correct_counts = np.zeros(10, dtype=np.int32)
    total_counts = np.zeros(10, dtype=np.int32)

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == pred_label:
            correct_counts[true_label] += 1
        total_counts[true_label] += 1

    ratios = correct_counts / total_counts

    for i, ratio in enumerate(ratios):
        print(
            f"Клас {i} ({classes[i]}): правильно класифіковано {correct_counts[i]} з {total_counts[i]} ({ratio:.2%})"
        )


def calculate_percent_right_base(test, labels, model):
    y_pred_probs = model.predict(test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(labels, axis=1)

    correct_counts = np.zeros(10, dtype=np.int32)
    total_counts = np.zeros(10, dtype=np.int32)

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == pred_label:
            correct_counts[true_label] += 1
        total_counts[true_label] += 1

    ratios = correct_counts / total_counts

    for i, ratio in enumerate(ratios):
        print(
            f"Клас {i} ({classes[i]}): правильно класифіковано {correct_counts[i]} з {total_counts[i]} ({ratio:.2%})"
        )
