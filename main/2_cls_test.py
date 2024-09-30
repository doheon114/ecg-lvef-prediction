from sktime.classification.deep_learning import InceptionTimeClassifier, MVTSTransformerClassifier
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, precision_recall_curve
from sklearn.model_selection import KFold
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
import os
import json
from datetime import datetime
import dask
import dask.dataframe

import warnings

warnings.filterwarnings("ignore")

# custom
from utils_2 import evaluate_metrics, vis_history

# hyperparameters
th = 0.4  # 기준값을 0.5로 설정
n_epochs = 200
x_shape = (1000, 4  )
k_folds = 5

use_residual = False
use_bottleneck = False
depth = 6
kernel_size = 20
n_filters = 32
batch_size = 16

# data
with open("data/processed.pkl", "rb") as f:
    data = pickle.load(f)
    X_train = data["train"]["x"]
    y_train = data["train"]["y"]
    X_int = data["int test"]["x"]
    y_int = data["int test"]["y"]
    X_ext = data["ext test"]["x"]
    y_ext = data["ext test"]["y"]

# 레이블을 이진화
y_train_binary = (y_train >= th).astype(np.int64)
y_int_binary = (y_int >= th).astype(np.int64)
y_ext_binary = (y_ext >= th).astype(np.int64)

# y 라벨 비율 계산 및 시각화 함수 정의
def plot_label_distribution(y_data, dataset_name, save_path):
    label_counts = dict(Counter(y_data))
    plt.figure(figsize=(6, 4))
    bars = plt.bar(label_counts.keys(), label_counts.values(), color=['blue', 'orange'])
    plt.xticks(ticks=[0, 1], labels=["LVSD", "No LVSDS"])
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title(f'Label Distribution in {dataset_name}')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), ha='center', va='bottom')
    plt.savefig(save_path)
    plt.close()

# 각 데이터셋의 라벨 비율 시각화 및 저장
plot_label_distribution(y_train_binary, 'y_train', "results/y_train_labels.png")
plot_label_distribution(y_int_binary, 'y_int', "results/y_int_labels.png")
plot_label_distribution(y_ext_binary, 'y_ext', "results/y_ext_labels.png")

def ensemble_predict(models, X_data, voting='soft'):
    if voting == 'soft':
        y_preds = np.array([model.predict(X_data) for model in models])
        y_prob_avg = np.mean(y_preds, axis=0)
        y_pred = np.argmax(y_prob_avg, axis=1)
    else:
        y_preds = np.array([np.argmax(model.predict(X_data), axis=1) for model in models])
        y_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=y_preds)
    return y_pred

def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        cmap=plt.cm.Blues,
    )
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def train_cls_with_ensemble(base):
    times = datetime.today().strftime("%Y%m%d_%H:%M:%S")
    class_names = ["LVSD", "No LVSD"]
    models = []

    for lr in [0.000005]:
        PATH = f"results/cls/{times}_{base}_clip0.3/"
        os.makedirs(PATH, exist_ok=True)

        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=0)
        fold_no = 1

        for train_index, val_index in kfold.split(X_train):
            print(f"Training on fold {fold_no}...")
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train_binary[train_index], y_train_binary[val_index]

            plot_label_distribution(y_train_fold, f'y_train (Fold {fold_no})', f"results/fold/y_train_labels_{fold_no}.png")
            plot_label_distribution(y_val_fold, f'y_val (Fold {fold_no})', f"results/fold/y_val_labels_{fold_no}.png")

            class_weight = compute_class_weight('balanced', classes=np.unique(y_train_fold), y=y_train_fold)
            class_weight_dict = dict(enumerate(class_weight))

            clf = InceptionTimeClassifier(verbose=True,
                                        n_epochs = 200,
                                        kernel_size=kernel_size, 
                                        n_filters=n_filters, 
                                        use_residual=use_residual,
                                        use_bottleneck=use_bottleneck, 
                                        depth=depth, 
                                        loss = 'categorical_crossentropy',
                                        random_state=0)
            
            clf.fit(X_train_fold, y_train_fold)

            models.append(clf)
            # clf.save_weights(PATH + f"fold_{fold_no}/model_weights.h5")

            roc_data = {}
            prc_data = {}

            for X, y, dataset in [(X_int, y_int_binary, "int"), (X_ext, y_ext_binary, "ext")]:
                y_pred = clf.predict(X)
                y_pred = clf.predict(X)
                if y_pred.ndim == 1:  # y_pred가 1차원인 경우
                    y_prob = y_pred  # 직접적으로 사용
                else:  # y_pred가 2차원인 경우
                    y_prob = y_pred[:, 1]  # 클래스 확률을 추출
                if y_pred.ndim == 1:  # y_pred is 1D
                    y_pred_bi = (y_pred >= 0.5).astype(np.int64)  # Apply your threshold
                else:  # y_pred is 2D
                    y_pred_bi = np.argmax(y_pred, axis=1)
                y = (y >= th).astype(np.int64)

                # Confusion matrix
                confusion_matrix_path = PATH + f"fold_{fold_no}/{dataset}_CM.png"  # Path for confusion matrix
                plot_confusion_matrix(y, y_pred_bi, class_names, f'Confusion Matrix - {dataset} (Fold {fold_no})', confusion_matrix_path)
                # ROC 곡선 데이터 저장
                fpr, tpr, _ = roc_curve(y, y_prob)
                roc_auc = auc(fpr, tpr)
                roc_data[dataset] = (fpr, tpr, roc_auc)

                # Precision-Recall Curve 데이터 저장
                precision, recall, _ = precision_recall_curve(y, y_prob)
                prc_auc = auc(recall, precision)
                prc_data[dataset] = (precision, recall, prc_auc)

                metrics = evaluate_metrics(y, y_pred_bi, y_pred)
                print(f"Dataset: {dataset}, Fold: {fold_no}, Learning Rate: {lr}")
                print(f"Accuracy: {metrics['accuracy']}")
                print(f"Sensitivity (Recall): {metrics['sensitivity']}")
                print(f"Specificity: {metrics['specificity']}")
                print(f"F1-score: {metrics['f1_score']}")
                print(f"AUROC: {metrics['auroc']}")
                print(f"AUPRC: {metrics['auprc']}")

                with open(PATH + f"fold_{fold_no}/history.json", "a") as f:
                    json.dump({dataset: metrics}, f, indent=4)

            plot_combined_roc_curves(roc_data, prc_data, PATH + f"fold_{fold_no}/")

            fold_no += 1

    for dataset, X, y in [("int", X_int, y_int_binary), ("ext", X_ext, y_ext_binary)]:
        y_pred_soft = ensemble_predict(models, X, voting='soft')
        y_pred_hard = ensemble_predict(models, X, voting='hard')

        print(f"\n=== {dataset.upper()} DATASET ENSEMBLE RESULTS ===")
        for voting_type, y_pred in zip(['soft', 'hard'], [y_pred_soft, y_pred_hard]):
            metrics = evaluate_metrics(y, y_pred, y_pred)
            print(f"Accuracy: {metrics['accuracy']}")
            print(f"Sensitivity (Recall): {metrics['sensitivity']}")
            print(f"Specificity: {metrics['specificity']}")
            print(f"F1-score: {metrics['f1_score']}")
            print(f"AUROC: {metrics['auroc']}")
            print(f"AUPRC: {metrics['auprc']}")

            with open(PATH + f"{voting_type}_history.json", "a") as f:
                json.dump({dataset: metrics}, f, indent=4)

            plot_confusion_matrix(
                y,
                y_pred,
                class_names,
                f'Confusion Matrix - {dataset} Ensemble {voting_type.capitalize()} Voting',
                PATH + f"ensemble_{dataset}_{voting_type}_confusion_matrix.png"
            )



def plot_combined_roc_curves(roc_data, prc_data, PATH):
    plt.figure(figsize=(16, 6))
    
    # ROC Curve 플롯
    plt.subplot(1, 2, 1)
    for dataset, (fpr, tpr, roc_auc) in roc_data.items():
        plt.plot(fpr, tpr, lw=2, label=f'{dataset} (AUROC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate [%]')
    plt.ylabel('True Positive Rate [%]')
    plt.title('ROC Curve for Internal and External Datasets')
    plt.legend(loc='lower right')
    
    # Precision-Recall Curve 플롯
    plt.subplot(1, 2, 2)
    for dataset, (precision, recall, prc_auc) in prc_data.items():
        plt.plot(recall, precision, lw=2, label=f'{dataset} (AUPRC = {prc_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall [%]')
    plt.ylabel('Precision [%]')
    plt.title('Precision-Recall Curve for Internal and External Datasets')
    plt.legend(loc='lower left')
    
    plt.savefig(PATH + "combined_curves.png")
    plt.show()


if __name__ == "__main__":
    train_cls_with_ensemble(f"{x_shape[0], x_shape[1]}_{th})")