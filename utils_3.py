from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, 
    precision_recall_curve, auc, confusion_matrix
)

import numpy as np
import os
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

def evaluate_metrics_multiclass(y_true, y_pred, y_prob):
    """
    다양한 평가 메트릭을 계산합니다.
    Parameters:
    y_true (list or array): 실제 레이블.
    y_pred (list or array): 예측된 레이블.
    y_prob (list or array): 예측된 확률.
    Returns:
    dict: 정확도, 재현율, 정밀도, 특이도, F1 점수, AUROC, AUPRC를 포함한 사전.
    """
    # y_true, y_pred, y_prob이 1D 배열인지 확인
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    # 다중 클래스 분류의 경우, y_prob는 (n_samples, n_classes) 형태의 2D 배열이어야 합니다.
    if y_prob.ndim == 1:
        raise ValueError("다중 클래스 분류의 경우, y_prob는 (n_samples, n_classes) 형태의 2D 배열이어야 합니다.")

    # 혼동 행렬 요소 계산
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # 다중 클래스 분류의 경우, 혼동 행렬은 2x2보다 큽니다.
    # 각 클래스의 특이도를 계산합니다.
    specificity_list = []
    for i in range(conf_matrix.shape[0]):
        tn = np.sum(conf_matrix) - np.sum(conf_matrix[i, :]) - np.sum(conf_matrix[:, i]) + conf_matrix[i, i]
        fp = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
        specificity = tn / (tn + fp) if tn + fp > 0 else 0
        specificity_list.append(specificity)
    specificity = np.mean(specificity_list)

    # 메트릭 계산
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted')  # 다중 클래스의 가중 평균 사용
    precision = precision_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # 다중 클래스 분류를 위한 AUROC 계산
    auroc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')

    # 다중 클래스 분류를 위한 AUPRC 계산
    num_classes = y_prob.shape[1]
    auprc_list = []
    for i in range(num_classes):
        precision_vals, recall_vals, _ = precision_recall_curve(y_true == i, y_prob[:, i])
        auprc = auc(recall_vals, precision_vals)
        auprc_list.append(auprc)
    auprc = np.mean(auprc_list)

    return {
        'accuracy': np.round(accuracy, 4),
        'sensitivity': np.round(recall, 4),
        'precision': np.round(precision, 4),
        'specificity': np.round(specificity, 4),
        'f1_score': np.round(f1, 4),
        'auroc': np.round(auroc, 4),
        'auprc': np.round(auprc, 4)
    }



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
