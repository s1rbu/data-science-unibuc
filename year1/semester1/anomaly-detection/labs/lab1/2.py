from pyod.models.knn import KNN
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, roc_curve, RocCurveDisplay

def balanced_accuracy(conf_mx):
    true_neg, false_pos, false_neg, true_pos = conf_mx.ravel()
    print(f"TN = {true_neg}, FP = {false_pos}, FN = {false_neg}, TP = {true_pos}")

    true_pos_rate = true_pos / (true_pos + false_neg)
    true_neg_rate = true_neg / (true_neg + false_pos)

    return (true_pos_rate + true_neg_rate)/2


def plot_roc_curve(fpr_, tpr_):
    plt.plot(fpr_, tpr_, label="ROC Curve")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()


def plot_different_contaminations(X_train, X_test, y_test):
    contamination_set = [0.05, 0.1, 0.2, 0.5]

    for c in contamination_set:
        model = KNN(contamination=c)
        model.fit(X_train)

        y_test_pred = model.predict(X_test)
        y_test_scores = model.decision_function(X_test)

        cf_matrix = confusion_matrix(y_test, y_test_pred)
        bal_acc = balanced_accuracy(cf_matrix)
        fpr, tpr, _ = roc_curve(y_test, y_test_scores)

        fig, axes = plt.subplots(1, 2)
        fig.suptitle(f"KNN (contamination = {c}) | Balanced Accuracy = {bal_acc: .2f}")

        ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, display_labels=['Normal', 'Outlier'], ax=axes[0])
        axes[0].set_title("Confusion Matrix")

        axes[1].plot(fpr, tpr, label="ROC Curve")
        axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("ROC Curve")
        axes[1].legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    with np.load("data/dataset.npz") as data:
        X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

    model = KNN(contamination=0.1)
    model.fit(X_train)

    y_train_pred = model.labels_
    y_test_pred = model.predict(X_test)

    cf_matrix = confusion_matrix(y_test, y_test_pred)
    print("Confusion matrix: \n", cf_matrix)

    ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, display_labels=['Normal', 'Outlier'])
    plt.show()

    print("My Balanced accuracy: ", balanced_accuracy(cf_matrix))
    print("Sklearn Balanced accuracy: ", balanced_accuracy_score(y_test, y_test_pred))

    y_test_scores = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_test_scores)
    plot_roc_curve(fpr, tpr)

    RocCurveDisplay.from_predictions(y_test, y_test_scores)
    plt.show()

    plot_different_contaminations(X_train, X_test, y_test)