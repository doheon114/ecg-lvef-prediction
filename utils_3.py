from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, 
    precision_recall_curve, auc, confusion_matrix
)

import numpy as np
import os
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

def plot_combined_roc_curves(roc_data, PATH):
    plt.figure(figsize=(8, 6))
    
    for dataset, (fpr, tpr, roc_auc) in roc_data.items():
        plt.plot(fpr, tpr, lw=2, label=f'{dataset} (AUROC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate [%]')
    plt.ylabel('True Positive Rate [%]')
    plt.title('ROC Curve for Internal and External Datasets')
    plt.legend(loc='lower right')
    
    plt.savefig(PATH + "combined_ROC.png")
    plt.show()
def evaluate_metrics(y_true, y_pred, y_prob):
    """
    Computes various evaluation metrics.
    Parameters:
    y_true (list or array): True labels.
    y_pred (list or array): Predicted labels.
    y_prob (list or array): Predicted probabilities.
    Returns:
    dict: Dictionary with accuracy, recall, precision, specificity, f1 score, auroc, and auprc.
    """
    # Ensure y_true, y_pred, y_prob are 1D arrays
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    # For multi-class classification, y_prob should be a 2D array with shape (n_samples, n_classes)
    if y_prob.ndim == 1:
        raise ValueError("For multi-class classification, y_prob should be a 2D array with shape (n_samples, n_classes).")

    # Calculate confusion matrix elements
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # For multi-class classification, confusion matrix will be larger than 2x2
    # Here, we will compute the specificity for each class
    specificity_list = []
    for i in range(conf_matrix.shape[0]):
        tn = np.sum(conf_matrix) - np.sum(conf_matrix[i, :]) - np.sum(conf_matrix[:, i]) + conf_matrix[i, i]
        fp = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
        specificity = tn / (tn + fp) if tn + fp > 0 else 0
        specificity_list.append(specificity)
    specificity = np.mean(specificity_list)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted')  # Use weighted average for multi-class
    precision = precision_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Calculate AUROC for multi-class classification
    auroc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')

    # Calculate AUPRC for multi-class classification
    num_classes = y_prob.shape[1]
    auprc_list = []
    for i in range(num_classes):
        precision_vals, recall_vals, _ = precision_recall_curve(y_true == i, y_prob[:, i])
        auprc = auc(recall_vals, precision_vals)
        auprc_list.append(auprc)
    auprc = np.mean(auprc_list)

    return {
        'accuracy': np.round(accuracy, 4),
        'recall': np.round(recall, 4),
        'precision': np.round(precision, 4),
        'specificity': np.round(specificity, 4),
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