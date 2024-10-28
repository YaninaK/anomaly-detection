import matplotlib.pyplot as plt
import numpy as np


def plot_loss(history):
    """
    Отрисовывает график обучения модели LSTM в tensorflow, сопоставляя значение
    функции потерь на обучающем и валидационном датасетах в разрезе эпох.

    """

    plt.semilogy(history.epoch, history.history["loss"], label="Train")
    plt.semilogy(
        history.epoch, history.history["val_loss"], label="Valid", linestyle="--"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM model Loss")
    plt.legend()


def plot_metrics(history):
    """
    Отрисовывает графики изменения метрик MAE, MAPE, MSE и RMSE модели в tensorflow,
    сопоставляя их значения на обучающем и валидационном датасетах в разрезе эпох.
    """
    metrics = ["mae", "mape", "mse", "rmse"]
    plt.figure(figsize=(10, 8))
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], label="Train")
        plt.plot(
            history.epoch, history.history["val_" + metric], linestyle="--", label="Val"
        )
        plt.xlabel("Epoch")
        plt.ylabel(name)

        plt.legend()
