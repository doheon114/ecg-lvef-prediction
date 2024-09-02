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
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# custom
from utils_3 import evaluate_metrics, vis_history

# hyperparameters
th_low = 0.4  # 낮은 기준값
th_high = 0.5  # 높은 기준값
n_epochs = 200
x_shape = (1000, 4)  # 12-lead + rhythm thread
k_folds = 5

use_residual=True
use_bottleneck=True
depth=6
kernel_size=10
n_filters=32
batch_size=32

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

# 0.4과 0.5 기준으로 삼중 분류 레이블 생성
def to_triple_class(y, low_th=0.4, high_th=0.5):
    return np.where(y < low_th, 0, np.where(y >= high_th, 2, 1))

y_train_triple = to_triple_class(y_train, low_th=th_low, high_th=th_high)
y_int_triple = to_triple_class(y_int, low_th=th_low, high_th=th_high)
y_ext_triple = to_triple_class(y_ext, low_th=th_low, high_th=th_high)

# y_train 라벨 비율 계산 및 시각화
unique, counts = np.unique(y_train_triple, return_counts=True)
label_counts = dict(zip(unique, counts))

plt.figure(figsize=(6, 4))
plt.bar(label_counts.keys(), label_counts.values(), color=['blue', 'orange', 'green'])
plt.xticks(ticks=[0, 1, 2], labels=["EF<40%", "40%<=EF<50%", "EF>=50%"])
plt.xlabel('Classes')
plt.ylabel('Number of Samples')
plt.title('Label Distribution in y_train (Triple Classification)')
plt.savefig("/home/work/.LVEF/ecg-lvef-prediction/results/cls/labels_triple.png")
plt.show()

def train_cls(base):
    times = datetime.today().strftime("%Y%m%d_%H:%M:%S")
    class_names = ["EF<40%", "40%<=EF<50%", "EF>=50%"]

    for lr in [0.000005]:
        PATH = f"results/cls/{times}_{base}_clip0.3/{lr}/"
        os.makedirs(PATH, exist_ok=True)

        # KFold 설정
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=0)
        fold_no = 1

        for train_index, val_index in kfold.split(X_train):
            print(f"Training on fold {fold_no}...")
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train_triple[train_index], y_train_triple[val_index]

            clf = InceptionTimeClassifier(verbose=True, kernel_size=kernel_size, use_residual=use_residual, use_bottleneck=use_bottleneck, depth=depth, n_filters=n_filters, random_state=0).build_model(
                input_shape=(x_shape[0], x_shape[1]), n_classes=3
            )

            clf.compile(optimizer=Adam(learning_rate=lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            clf.summary()

            # Training the model
            history = clf.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold), epochs=n_epochs, batch_size=batch_size, shuffle=True)

            vis_history(history, PATH + f"fold_{fold_no}/", lr)  # 학습 히스토리 시각화 및 저장

            # ROC 곡선 및 평가 메트릭 저장
            roc_data = {}

            for X, y, dataset in [(X_int, y_int_triple, "int"), (X_ext, y_ext_triple, "ext")]:
                y_pred = clf.predict(X)
                y_prob = y_pred  # 전체 클래스에 대한 확률
                y_pred_bi = np.argmax(y_pred, axis=1)

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
                roc_data[dataset] = {}
                for i in range(3):
                    fpr, tpr, _ = roc_curve(y == i, y_prob[:, i])
                    roc_auc = auc(fpr, tpr)
                    roc_data[dataset][f"Class {i}"] = (fpr, tpr, roc_auc)

                # 평가 메트릭 계산
                metrics = evaluate_metrics(y, y_pred_bi, y_pred)
                print(f"Dataset: {dataset}, Fold: {fold_no}, Learning Rate: {lr}")
                print(f"Accuracy: {metrics['accuracy']}")
                print(f"Sensitivity (Recall): {metrics['recall']}")
                print(f"Specificity: {metrics['specificity']}")
                print(f"F1-score: {metrics['f1_score']}")
                print(f"AUROC: {metrics['auroc']}")
                print(f"AUPRC: {metrics['auprc']}")

                # 평가 메트릭을 history.json 파일에 기록
                with open(PATH + f"fold_{fold_no}/history.json", "a") as f:
                    json.dump({dataset: metrics}, f, indent=4)

            # Combined ROC curves 시각화
            plot_combined_roc_curves(roc_data, PATH + f"fold_{fold_no}/")

            fold_no += 1

def plot_combined_roc_curves(roc_data, PATH):
    plt.figure(figsize=(8, 6))
    
    for dataset, classes in roc_data.items():
        for class_name, (fpr, tpr, roc_auc) in classes.items():
            plt.plot(fpr, tpr, lw=2, label=f'{dataset} - {class_name} (AUROC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate [%]')
    plt.ylabel('True Positive Rate [%]')
    plt.title('ROC Curve for Internal and External Datasets')
    plt.legend(loc='lower right')
    
    plt.savefig(PATH + "combined_ROC.png")
    plt.show()

if __name__ == "__main__":
    train_cls(f"{x_shape[0], x_shape[1]}_{th_low}_{th_high}")
