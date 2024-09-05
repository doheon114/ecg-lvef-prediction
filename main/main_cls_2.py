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

warnings.filterwarnings("ignore")

# custom
from utils_2 import evaluate_metrics, vis_history

# hyperparameters
th = 0.4  # 기준값을 0.5로 설정
n_epochs = 300
x_shape = (1000, 4)  
k_folds = 5

use_residual=False
use_bottleneck=False
depth=6
kernel_size=20
n_filters=32 
batch_size=16

# data
with open("data/processed.pkl", "rb") as f:
    data = pickle.load(f)
    num = x_shape[1]
    X_train = data["train"]["x"]
    y_train = data["train"]["y"]
    X_int = data["int test"]["x"]
    y_int = data["int test"]["y"]
    X_ext = data["ext test"]["x"]
    y_ext = data["ext test"]["y"]

# 레이블을 이진화한 후의 데이터셋들
y_train_binary = (y_train >= th).astype(np.int64)
y_int_binary = (y_int >= th).astype(np.int64)
y_ext_binary = (y_ext >= th).astype(np.int64)

# y 라벨 비율 계산 및 시각화 함수 정의
def plot_label_distribution(y_data, dataset_name, save_path):
    label_counts = dict(Counter(y_data))
    plt.figure(figsize=(6, 4))
    bars = plt.bar(label_counts.keys(), label_counts.values(), color=['blue', 'orange'])
    plt.xticks(ticks=[0, 1], labels=["EF<40%", "EF>40%"])
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title(f'Label Distribution in {dataset_name}')
    # 막대 위에 개수를 추가
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), ha='center', va='bottom')
    
    plt.savefig(save_path)
    plt.close()

# 각 데이터셋의 라벨 비율 시각화 및 저장
plot_label_distribution(y_train_binary, 'y_train', "/home/work/.LVEF/ecg-lvef-prediction/results/y_train_labels.png")
plot_label_distribution(y_int_binary, 'y_int', "/home/work/.LVEF/ecg-lvef-prediction/results/y_int_labels.png")
plot_label_distribution(y_ext_binary, 'y_ext', "/home/work/.LVEF/ecg-lvef-prediction/results/y_ext_labels.png")

# k-fold를 사용한 훈련 과정에서도 각 fold의 validation set에 대한 라벨 분포를 시각화할 수 있습니다.
def train_cls(base):
    times = datetime.today().strftime("%Y%m%d_%H:%M:%S")
    class_names = ["EF<40%", "EF>40%"]

    int_accuracies = []  # int 데이터셋의 accuracy를 저장할 리스트
    ext_accuracies = []  # ext 데이터셋의 accuracy를 저장할 리스트

    for lr in [0.000005]:
        PATH = f"results/cls/{times}_{base}_clip0.3/{lr}/"
        os.makedirs(PATH, exist_ok=True)

        # KFold 설정
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=0)
        fold_no = 1

        for train_index, val_index in kfold.split(X_train):
            print(f"Training on fold {fold_no}...")
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train_binary[train_index], y_train_binary[val_index]

            # validation set의 라벨 분포 시각화 및 저장
            plot_label_distribution(y_train_fold, f'y_train (Fold {fold_no})', f"/home/work/.LVEF/ecg-lvef-prediction/results/fold/y_train_labels_{fold_no}.png")
            plot_label_distribution(y_val_fold, f'y_val (Fold {fold_no})', f"/home/work/.LVEF/ecg-lvef-prediction/results/fold/y_val_labels_{fold_no}.png")

            # 클래스 가중치 계산
            class_weight = compute_class_weight('balanced', classes=np.unique(y_train_fold), y=y_train_fold)
            class_weight_dict = dict(enumerate(class_weight))
            
            # 클래스 가중치 출력 추가
            print(f"Fold {fold_no} - Class Weights: {class_weight_dict}")

            # 모델 생성 및 컴파일
            clf = InceptionTimeClassifier(verbose=True,
                                        kernel_size=kernel_size, 
                                        n_filters=n_filters, 
                                        use_residual=use_residual,
                                        use_bottleneck=use_bottleneck, 
                                        depth=depth, 
                                        random_state=0).build_model(
                input_shape=(1000, 4), n_classes=2
            )

            clf.compile(optimizer=Adam(learning_rate=lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            clf.summary()

            # 가중치를 이용해 모델 훈련
            history = clf.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold), epochs=n_epochs, batch_size=batch_size, shuffle=True, class_weight=class_weight_dict)

            vis_history(history, PATH + f"fold_{fold_no}/", lr)  # 학습 히스토리 시각화 및 저장

            # ROC 곡선 및 평가 메트릭 저장
            roc_data = {}
            prc_data = {}

            for X, y, dataset in [(X_int, y_int_binary, "int"), (X_ext, y_ext_binary, "ext")]:
                y_pred = clf.predict(X)
                y_prob = y_pred[:, 1]  # 클래스 1에 대한 확률
                y_pred_bi = np.argmax(y_pred, axis=1)
                y = (y >= th).astype(np.int64)

                # Confusion matrix
                ConfusionMatrixDisplay.from_predictions(
                    y,
                    y_pred_bi,
                    display_labels=class_names,
                    cmap=plt.cm.Blues,
                )
                plt.savefig(PATH + f"fold_{fold_no}/{dataset}_CM.png")
                plt.close()

                # ROC 곡선 데이터 저장
                fpr, tpr, _ = roc_curve(y, y_prob)
                roc_auc = auc(fpr, tpr)
                roc_data[dataset] = (fpr, tpr, roc_auc)

                # Precision-Recall Curve 데이터 저장
                precision, recall, _ = precision_recall_curve(y, y_prob)
                prc_auc = auc(recall, precision)
                prc_data[dataset] = (precision, recall, prc_auc)

                # 평가 메트릭 계산
                metrics = evaluate_metrics(y, y_pred_bi, y_pred)
                print(f"Dataset: {dataset}, Fold: {fold_no}, Learning Rate: {lr}")
                print(f"Accuracy: {metrics['accuracy']}")
                print(f"Sensitivity (Recall): {metrics['sensitivity']}")
                print(f"Specificity: {metrics['specificity']}")
                print(f"F1-score: {metrics['f1_score']}")
                print(f"AUROC: {metrics['auroc']}")
                print(f"AUPRC: {metrics['auprc']}")

                # 평가 메트릭을 history.json 파일에 기록
                with open(PATH + f"fold_{fold_no}/history.json", "a") as f:
                    json.dump({dataset: metrics}, f, indent=4)

                # accuracy 저장
                if dataset == "int":
                    int_accuracies.append(metrics['accuracy'])
                elif dataset == "ext":
                    ext_accuracies.append(metrics['accuracy'])

            # Combined ROC curves 시각화
            plot_combined_roc_curves(roc_data, prc_data, PATH + f"fold_{fold_no}/")

            fold_no += 1

    # 전체 폴드에서의 int, ext accuracy 평균 출력
    print(f"\nMean accuracy for int dataset across {k_folds} folds: {np.mean(int_accuracies):.4f}")
    print(f"Mean accuracy for ext dataset across {k_folds} folds: {np.mean(ext_accuracies):.4f}")


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
    train_cls(f"{x_shape[0], x_shape[1]}_{th})")
