import numpy as np
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data
from sklearn.metrics import balanced_accuracy_score

if __name__ == "__main__":
    X, _, y, _ = generate_data(n_train=1000, n_test=0, n_features=1, contamination=0.1)
    # np.savez("data/unidimensional_dataset.npz", X=X, y=y)

    mean = np.mean(X)
    print("Mean: ", mean)
    std = np.std(X)
    print("Standard Deviation: ", std)
    z_scores = np.abs((X - mean) / std)
    print("Z scores sample: ", z_scores[:5])

    threshold = np.quantile(z_scores, 0.9)  # 1-contamination_rate
    print("Threshold: ", threshold)

    y_pred = (z_scores > threshold).astype(int)

    balanced_accuracy = balanced_accuracy_score(y, y_pred)
    print("Balanced Accuracy: ", balanced_accuracy)
