from pyod.utils.data import generate_data
import matplotlib.pyplot as plt
import numpy as np

def plot_data(X_train_, X_test, y_train, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, X, y, title in zip(
        axes, # [axes[0], axes[1]]
        [X_train_, X_test],
        [y_train, y_test],
        ["Training Data", "Testing Data"]
    ):
        ax.scatter(X[y == 0, 0], X[y == 0, 1], c="green", label="Normal", marker="o", edgecolors="black", alpha=0.5)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c="red", label="Outlier", marker="^", alpha=0.7)
        ax.set_title(title)
        ax.legend()

    plt.suptitle("Normal vs Outlier Data")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = generate_data(
        n_train=400, n_test=100, n_features=2, contamination=0.1
    )

    # np.savez("data/dataset.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    # with np.load("data/dataset.npz") as data:
    #     X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

    plot_data(X_train, X_test, y_train, y_test)