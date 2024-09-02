from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, 
    precision_recall_curve, auc, confusion_matrix
)

import numpy as np
import os
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
def evaluate_metrics(y_true, y_pred, y_prob):
    """
    Computes various evaluation metrics.
    Parameters:
    y_true (list or array): True binary labels.
    y_pred (list or array): Predicted binary labels.
    y_prob (list or array): Predicted probabilities for the positive class.
    Returns:
    dict: Dictionary with accuracy, recall, precision, specificity, f1 score, auroc, and auprc.
    """
    # Ensure y_true, y_pred, y_prob are 1D arrays
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    
    # For binary classification, y_prob should be a 2D array with shape (n_samples, 2)
    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]  # 양성 클래스(EF<40%)에 대한 확률로 수정

    y_prob = np.ravel(y_prob)

    # Calculate confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate accuracy, recall, precision, f1 score
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    specificity = tn / (tn + fp)
    f1 = f1_score(y_true, y_pred)

    # Calculate AUROC
    auroc = roc_auc_score(y_true, y_prob)

    # Calculate AUPRC
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(recall_vals, precision_vals)

    return {
        'accuracy': np.round(accuracy, 4),
        'sensitivity': np.round(recall, 4),
        'precision': np.round(precision, 4),
        'specificity':  np.round(specificity, 4),
        'f1_score': np.round(f1, 4),
        'auroc': np.round(auroc, 4),
        'auprc': np.round(auprc, 4)
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