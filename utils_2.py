from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, 
    precision_recall_curve, auc, confusion_matrix
)

import numpy as np
import os
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as pltfrom sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, 
    precision_recall_curve, auc, confusion_matrix
)

def evaluate_metrics_multiclass(y_true, y_pred, y_prob):
    """
    Computes various evaluation metrics for multiclass classification.
    Parameters:
    y_true (list or array): True labels.
    y_pred (list or array): Predicted labels.
    y_prob (list or array): Predicted probabilities for each class.
    Returns:
    dict: Dictionary with accuracy, macro/micro/weighted precision, recall, f1 score, and AUROC.
    """
    # Ensure y_true, y_pred are 1D arrays
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate macro, micro, weighted recall, precision, f1 score
    recall_macro = recall_score(y_true, y_pred, average='macro')
    precision_macro = precision_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    # Calculate AUROC (for multiclass, one-vs-rest)
    auroc = roc_auc_score(y_true, y_prob, multi_class='ovr')

    return {
        'accuracy': np.round(accuracy, 4),
        'recall_macro': np.round(recall_macro, 4),
        'precision_macro': np.round(precision_macro, 4),
        'f1_macro': np.round(f1_macro, 4),
        'auroc': np.round(auroc, 4)
    }

import matplotlib.pyplot as plt
def vis_history(history, PATH, lr) :
    plt.figure(figsize=(18,6))

    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], color="blue", label = "Loss")
    plt.plot(history.history['val_loss'], color="orange", label = "Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("Number of Epochs")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], color="green", label = "Accuracy")
    plt.plot(history.history['val_accuracy'], color="red", label = "Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Number of Epochs")
    plt.legend()
      
    # 경로가 존재하지 않을 경우 생성
    os.makedirs(PATH, exist_ok=True)

    plt.tight_layout()
    plt.savefig(PATH + "history.png")