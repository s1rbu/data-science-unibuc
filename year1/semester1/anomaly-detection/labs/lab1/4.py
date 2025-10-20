import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score

def plot_original_dataset(y_):
    plt.scatter(y_[:,0], y_[:,1], color="green", marker="o", alpha=0.5, label="Pure Data")
    plt.scatter(mu[0], mu[1], color='red', marker='x', s=100, label='μ')
    plt.title("Original Dataset")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_contaminated_dataset(y_, y_true_):
    plt.scatter(y_[y_true_ == 0,0], y_[y_true_ == 0,1], color="green", marker="o", alpha=0.5, label="Normal")
    plt.scatter(y_[y_true_ == 1,0], y_[y_true_ == 1,1], color='red', marker="^", alpha=0.7, label="Outlier")
    plt.title("Contaminated Dataset")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(y_, predictions):
    plt.scatter(y_[predictions == 0,0], y_[predictions == 0,1], color="green", marker="o", alpha=0.5, label="TN")
    plt.scatter(y_[predictions == 1,0], y_[predictions == 1,1], color="red", marker="^", alpha=0.7, label="FN")
    plt.scatter(y_[predictions == 2,0], y_[predictions == 2,1], color="purple", marker="x", alpha=0.7, label="FP")
    plt.scatter(y_[predictions == 3,0], y_[predictions == 3,1], color="blue", marker="v", alpha=0.7, label="TP")
    plt.title("Predictions")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    mu = np.random.random(size=(2,))
    print("mu: ", mu)
    M = np.random.random(size=(2,2))

    sigma = np.dot(M, np.transpose(M))
    print("Sigma: ", sigma)

    L = cholesky(sigma)
    print("L: ", L)

    X = np.random.randn(1000, 2)
    y = X @ L.T + mu
    # print("y: ", y)

    # np.savez("data/multidimensional_dataset.npz", X=X, y=y)

    plot_original_dataset(y)

    y_true = np.zeros(1000)
    anomaly_idx = np.random.choice(1000, 100, replace=False)
    y_true[anomaly_idx] = 1.0

    y[anomaly_idx] += np.random.uniform(2, 4)

    plot_contaminated_dataset(y, y_true)

    y_mean = np.mean(y, axis=0)
    print("y_mean: ", y_mean)
    y_cov = np.cov(y, rowvar=False)
    inv_cov = np.linalg.inv(y_cov)

    diff = y - y_mean
    z_scores = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
    threshold = np.quantile(z_scores, 0.9) # 1-contamination

    y_pred = (z_scores > threshold).astype(int)

    y_combined = np.zeros_like(y_true)
    y_combined[(y_true == 0) & (y_pred == 0)] = 0  # TN
    y_combined[(y_true == 1) & (y_pred == 0)] = 1  # FN
    y_combined[(y_true == 0) & (y_pred == 1)] = 2  # FP
    y_combined[(y_true == 1) & (y_pred == 1)] = 3  # TP

    plot_predictions(y, y_combined)

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    print("Balanced Accuracy: ", bal_acc)