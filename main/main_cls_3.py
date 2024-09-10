from sktime.classification.deep_learning import InceptionTimeClassifier
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import json
import warnings
import os
from datetime import datetime
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, precision_recall_curve
from sklearn.model_selection import KFold
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

from utils_3 import evaluate_metrics_multiclass, vis_history

# 하이퍼파라미터
th = [0.4, 0.5]  # 임계값
n_epochs = 200
x_shape = (1000, 4)
k_folds = 5

use_residual = False
use_bottleneck = False
depth = 6
kernel_size = 20
n_filters = 32
batch_size = 16

# 데이터 로딩
with open("data/processed.pkl", "rb") as f:
    data = pickle.load(f)
    num = x_shape[1]
    X_train = data["train"]["x"]
    y_train = data["train"]["y"]
    X_int = data["int test"]["x"]
    y_int = data["int test"]["y"]
    X_ext = data["ext test"]["x"]
    y_ext = data["ext test"]["y"]

# Label Binarization and Multi-Class Conversion
def binarize_labels(y, thresholds):
    return np.digitize(y, bins=thresholds, right=True)

y_train_multi = binarize_labels(y_train, th)
y_int_multi = binarize_labels(y_int, th)
y_ext_multi = binarize_labels(y_ext, th)

# 레이블 이진화 및 다중 클래스 변환
def plot_label_distribution(y_data, dataset_name, save_path):
    label_counts = dict(Counter(y_data))
    all_classes = [0, 1, 2]
    counts = [label_counts.get(cls, 0) for cls in all_classes]
    
    plt.figure(figsize=(6, 4))
    bars = plt.bar(all_classes, counts, color=['blue', 'orange', 'green'])
    plt.xticks(ticks=all_classes, labels=[f'Class {i}' for i in all_classes])
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title(f'Label Distribution in {dataset_name}')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), ha='center', va='bottom')
    plt.savefig(save_path)
    plt.close()

# 각 데이터셋에 대해 레이블 분포 시각화
plot_label_distribution(y_train_multi, 'y_train', "/home/work/.LVEF/ecg-lvef-prediction/results/y_train_labels.png")
plot_label_distribution(y_int_multi, 'y_int', "/home/work/.LVEF/ecg-lvef-prediction/results/y_int_labels.png")
plot_label_distribution(y_ext_multi, 'y_ext', "/home/work/.LVEF/ecg-lvef-prediction/results/y_ext_labels.png")

# ROC & PRC 계산
def compute_roc_prc(y_true, y_pred_proba, n_classes):
    roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
    prc_auc = average_precision_score(y_true, y_pred_proba, average='weighted')

    # ROC 곡선 및 PR 곡선 계산
    fpr, tpr, _ = roc_curve(y_true.ravel(), y_pred_proba.ravel())
    precision, recall, _ = precision_recall_curve(y_true.ravel(), y_pred_proba.ravel())
    
    return roc_auc, prc_auc, fpr, tpr, precision, recall

def plot_combined_roc_curves(roc_data, prc_data, PATH):
    plt.figure(figsize=(16, 6))
    
    # ROC Curve 플롯
    plt.subplot(1, 2, 1)
    for dataset, (roc_auc, fpr, tpr) in roc_data.items():
        plt.plot(fpr, tpr, lw=2, label=f'{dataset} (AUROC = {roc_auc:.3f}, macro)')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate [%]')
    plt.ylabel('True Positive Rate [%]')
    plt.title('ROC Curve for Multi-Class Classification')
    plt.legend(loc='lower right')
    
    # Precision-Recall Curve 플롯
    plt.subplot(1, 2, 2)
    for dataset, (prc_auc, precision, recall) in prc_data.items():
        plt.plot(recall, precision, lw=2, label=f'{dataset} (AUPRC = {prc_auc:.3f}, macro)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall [%]')
    plt.ylabel('Precision [%]')
    plt.title('Precision-Recall Curve for Multi-Class Classification')
    plt.legend(loc='lower left')
    
    plt.savefig(PATH + "combined_curves.png")
    plt.show()

def ensemble_predict(models, X_data, voting='soft'):
    """
    앙상블 예측 함수: 하드 보팅 또는 소프트 보팅을 사용해 다수 모델의 예측을 결합
    Args:
        models (list): 훈련된 모델들의 리스트
        X_data (np.array): 예측할 데이터셋
        voting (str): 'hard' 또는 'soft' 보팅 방식 선택
    Returns:
        y_pred (np.array): 앙상블된 최종 예측 결과
        y_pred_proba (np.array): 앙상블된 확률 (소프트 보팅 시)
    """
    if voting == 'soft':
        # 소프트 보팅 - 각 모델의 예측 확률을 평균
        y_preds = np.array([model.predict(X_data) for model in models])
        y_prob_avg = np.mean(y_preds, axis=0)  # 각 클래스에 대한 확률의 평균
        y_pred = np.argmax(y_prob_avg, axis=1)  # 평균 확률에 따라 최종 클래스 예측
        return y_pred, y_prob_avg
    else:
        # 하드 보팅 - 각 모델의 예측 결과(클래스)를 다수결로 결정
        y_preds = np.array([np.argmax(model.predict(X_data), axis=1) for model in models])
        y_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=y_preds)
        return y_pred, None

# K-Fold 교차 검증 및 모델 저장, 평가 with Ensemble and per-fold results
def train_cls_with_ensemble(base, voting_type="soft"):
    times = datetime.today().strftime("%Y%m%d_%H:%M:%S")
    class_names = ["EF<40%", "40%≤EF<50%", "EF≥50%"]

    models = []  # List to store models from each fold

    for lr in [0.000005]:
        PATH = f"results/cls/{times}_{base}_clip0.3/{lr}/"
        os.makedirs(PATH, exist_ok=True)

        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=0)
        fold_no = 1

        for train_index, val_index in kfold.split(X_train):
            print(f"Training on fold {fold_no}...")
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train_multi[train_index], y_train_multi[val_index]

            plot_label_distribution(y_train_fold, f'y_train (Fold {fold_no})', f"/home/work/.LVEF/ecg-lvef-prediction/results/fold/y_train_labels_{fold_no}.png")
            plot_label_distribution(y_val_fold, f'y_val (Fold {fold_no})', f"/home/work/.LVEF/ecg-lvef-prediction/results/fold/y_val_labels_{fold_no}.png")

            # 클래스별 가중치 계산
            class_weight = compute_class_weight('balanced', classes=np.unique(y_train_fold), y=y_train_fold)
            class_weight_dict = dict(enumerate(class_weight))
            print(f"Fold {fold_no} - Class Weights: {class_weight_dict}")

            # Train a model for each fold
            clf = InceptionTimeClassifier(
                verbose=True,
                kernel_size=kernel_size,
                n_filters=n_filters,
                use_residual=use_residual,
                use_bottleneck=use_bottleneck,
                depth=depth,
                random_state=fold_no  # use fold number as random state
            ).build_model(input_shape=x_shape, n_classes=3)

            clf.compile(optimizer=Adam(learning_rate=lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            clf.summary()

            # Train the model
            history = clf.fit(
                X_train_fold, y_train_fold,
                validation_data=(X_val_fold, y_val_fold),
                epochs=n_epochs,
                batch_size=batch_size,
                shuffle=True,
                class_weight=class_weight_dict
            )

            # Save the model for this fold
            model_path = PATH + f"fold_{fold_no}/model_fold_{fold_no}.h5"
            os.makedirs(PATH + f"fold_{fold_no}/", exist_ok=True)
            clf.save(model_path)
            models.append(clf)  # Store the model in the list

            # Visualization and evaluation on each fold
            vis_history(history, PATH + f"fold_{fold_no}/", lr)

            # Evaluate and save metrics for each fold model
            evaluate_and_save_fold_metrics(clf, fold_no, PATH, lr, class_names)
            fold_no += 1

        # Once all folds are trained, create an ensemble model from the models
        print("\nCreating an ensemble model from all fold models...")

        # Evaluate the final ensemble on internal and external test sets
        evaluate_ensemble(models, lr, PATH, voting_type='soft')
        evaluate_ensemble(models, lr, PATH, voting_type='hard')


def evaluate_and_save_fold_metrics(model, fold_no, PATH, lr, class_names):
    """Evaluate the model for each fold and save its metrics."""
    
    roc_data = {}
    prc_data = {}

    for X, y_multi, dataset in [(X_int, y_int_multi, "Internal"), (X_ext, y_ext_multi, "External")]:
        y_pred_proba = model.predict(X)
        y_true_bin = label_binarize(y_multi, classes=np.arange(3))

        # Confusion Matrix
        ConfusionMatrixDisplay.from_predictions(
            y_multi,
            np.argmax(y_pred_proba, axis=1),
            display_labels=class_names,
            cmap=plt.cm.Blues,
        )
        plt.savefig(PATH + f"fold_{fold_no}/{dataset}_CM.png")
        plt.close()

        # ROC and Precision-Recall Data 계산
        roc_auc, prc_auc, fpr, tpr, precision, recall = compute_roc_prc(y_true_bin, y_pred_proba, n_classes=3)

        if dataset == "Internal":
            roc_data['Internal'] = (roc_auc, fpr, tpr)
            prc_data['Internal'] = (prc_auc, precision, recall)
        elif dataset == "External":
            roc_data['External'] = (roc_auc, fpr, tpr)
            prc_data['External'] = (prc_auc, precision, recall)

        # Evaluate fold-specific metrics and save them
        metrics = evaluate_metrics_multiclass(y_multi, np.argmax(y_pred_proba, axis=1), y_pred_proba, y_true_bin)
        print(f"Dataset: {dataset}, Fold: {fold_no}, Learning Rate: {lr}")
        print(f"Accuracy: {metrics['accuracy']}")
        print(f"Sensitivity (Recall): {metrics['sensitivity']}")
        print(f"Specificity: {metrics['specificity']}")
        print(f"F1-score: {metrics['f1_score']}")
        print(f"AUROC: {metrics['auroc']}")
        print(f"AUPRC: {metrics['auprc']}")

        # Save fold-specific metrics to a file
        with open(PATH + f"fold_{fold_no}/{dataset}_metrics.json", "a") as f:
            json.dump({dataset: metrics}, f, indent=4)

def evaluate_ensemble(models, lr, PATH, voting_type='soft'):
    """Function to evaluate the ensemble model on internal and external datasets."""
    
    class_names = ["EF<40%", "40%≤EF<50%", "EF≥50%"]
    roc_data = {}
    prc_data = {}

    for X, y_multi, dataset in [(X_int, y_int_multi, "Internal"), (X_ext, y_ext_multi, "External")]:
        y_pred, y_pred_proba = ensemble_predict(models, X, voting=voting_type)
        y_true_bin = label_binarize(y_multi, classes=np.arange(3))

        # Confusion Matrix
        ConfusionMatrixDisplay.from_predictions(
            y_multi,
            y_pred,
            display_labels=class_names,
            cmap=plt.cm.Blues,
        )
        plt.savefig(PATH + f"ensemble_{dataset}_CM.png")
        plt.close()

        if y_pred_proba is not None:
            # ROC and Precision-Recall Data 계산
            roc_auc, prc_auc, fpr, tpr, precision, recall = compute_roc_prc(y_true_bin, y_pred_proba, n_classes=3)

            if dataset == "Internal":
                roc_data['Internal'] = (roc_auc, fpr, tpr)
                prc_data['Internal'] = (prc_auc, precision, recall)
            elif dataset == "External":
                roc_data['External'] = (roc_auc, fpr, tpr)
                prc_data['External'] = (prc_auc, precision, recall)

            metrics = evaluate_metrics_multiclass(y_multi, y_pred, y_pred_proba, y_true_bin)
        else:
            # Cannot compute ROC and PR curves without probabilities
            metrics = evaluate_metrics_multiclass(y_multi, y_pred, None, y_true_bin)
        if voting_type == 'soft':
            print(f"Dataset: {dataset}, Ensemble Model")
            print(f"Accuracy: {metrics['accuracy']}")
            print(f"Sensitivity (Recall): {metrics['sensitivity']}")
            print(f"Specificity: {metrics['specificity']}")
            print(f"F1-score: {metrics['f1_score']}")
            print(f"AUROC: {metrics.get('auroc', 'N/A')}")
            print(f"AUPRC: {metrics.get('auprc', 'N/A')}")
        else: 
            print(f"Dataset: {dataset}, Ensemble Model")
            print(f"Accuracy: {metrics['accuracy']}")
            print(f"Sensitivity (Recall): {metrics['sensitivity']}")
            print(f"Specificity: {metrics['specificity']}")
            print(f"F1-score: {metrics['f1_score']}")

        # Save ensemble metrics to a file
        with open(PATH + f"ensemble_metrics_{dataset}_{voting_type}.json", "a") as f:
            json.dump({dataset: metrics}, f, indent=4)

    if voting_type == 'soft':
        # Plot combined ROC and PRC curves for ensemble
        plot_combined_roc_curves(roc_data, prc_data, PATH)
    else:
        print("ROC and PR curves cannot be plotted for hard voting as probabilities are not available.")

if __name__ == "__main__":
    train_cls_with_ensemble(f"{x_shape[0], x_shape[1]}_{th}", voting_type="soft")
