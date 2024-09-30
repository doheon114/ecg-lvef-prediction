import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, 
    precision_recall_curve, auc, confusion_matrix
)

def evaluate_metrics(y_true, y_pred, y_prob):
    # Ensure y_true, y_pred are 1D arrays
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]

    y_prob = np.ravel(y_prob)

    tp = np.sum((y_true == 0) & (y_pred == 0))  # True Positive
    fn = np.sum((y_true == 0) & (y_pred == 1))  # False Negative
    fp = np.sum((y_true == 1) & (y_pred == 0))  # False Positive
    tn = np.sum((y_true == 1) & (y_pred == 1))  # True Negative

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    auroc = roc_auc_score(y_true, y_prob)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(recall_vals, precision_vals)

    return {
        'accuracy': float(np.round(accuracy, 4)),
        'sensitivity': float(np.round(recall, 4)),
        'precision': float(np.round(precision, 4)),
        'specificity': float(np.round(specificity, 4)),
        'f1_score': float(np.round(f1, 4)),
        'auroc': float(np.round(auroc, 4)),
        'auprc': float(np.round(auprc, 4))
    }


def vis_history(history, PATH, lr):
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], color="blue", label="Loss")
    plt.plot(history.history['val_loss'], color="orange", label="Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("Number of Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], color="green", label="Accuracy")
    plt.plot(history.history['val_accuracy'], color="red", label="Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Number of Epochs")
    plt.legend()
      
    # 경로가 존재하지 않을 경우 생성
    os.makedirs(PATH, exist_ok=True)

    plt.tight_layout()
    plt.savefig(PATH + "history.png")
