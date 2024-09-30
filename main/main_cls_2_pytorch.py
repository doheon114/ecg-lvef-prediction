from tsai.basics import *

import torch
from torch.optim import Adam
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, precision_recall_curve

from tsai.inference import load_learner

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from collections import Counter
from datetime import datetime
import os
import json

loss_func = torch.nn.CrossEntropyLoss()  # 다중 클래스 분류에 적합한 손실 함수

# 하이퍼파라미터
th = 0.4  # 기준값 설정
n_epochs = 300
x_shape = (1000, 4)
k_folds = 5

batch_size = 16

# 데이터 불러오기
with open("data/processed.pkl", "rb") as f:
    data = pickle.load(f)
    X_train = data["train"]["x"]
    y_train = data["train"]["y"]
    X_int = data["int test"]["x"]
    y_int = data["int test"]["y"]
    X_ext = data["ext test"]["x"]
    y_ext = data["ext test"]["y"]

print(X_train.shape)
print(X_int.shape)
print(X_ext.shape)
# # 데이터 reshape
# X_train = X_train.reshape(X_train.shape[0], 4, 1000)  # X_train의 shape을 (n_samples, 4, 1000)으로 변환
# X_int = X_int.reshape(X_int.shape[0], 4, 1000)         # X_int의 shape을 (n_samples, 4, 1000)으로 변환
# X_ext = X_ext.reshape(X_ext.shape[0], 4, 1000)         # X_ext의 shape을 (n_samples, 4, 1000)으로 변환


# 이진 레이블로 변환
y_train_binary = (y_train >= th).astype(np.int64)
y_int_binary = (y_int >= th).astype(np.int64)
y_ext_binary = (y_ext >= th).astype(np.int64)



# y 라벨 비율 시각화 함수
def plot_label_distribution(y_data, dataset_name, save_path):
    label_counts = dict(Counter(y_data))
    plt.figure(figsize=(6, 4))
    bars = plt.bar(label_counts.keys(), label_counts.values(), color=['blue', 'orange'])
    plt.xticks(ticks=[0, 1], labels=["LVSD", "No LVSD"])
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

from torchsummary import summary

# 훈련 함수
def train_cls_with_ensemble():
    times = datetime.today().strftime("%Y%m%d_%H:%M:%S")
    class_names = ["LVSD", "No LVSD"]

    models = []

    # 결과 저장 경로 생성
    PATH = f"results/cls/{times}_InceptionTime/"
    os.makedirs(PATH, exist_ok=True)

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=0)
    fold_no = 1

    for train_index, val_index in kfold.split(X_train):
        print(f"Training on fold {fold_no}...")
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train_binary[train_index], y_train_binary[val_index]

        plot_label_distribution(y_train_fold, f'y_train (Fold {fold_no})', f"results/fold/y_train_labels_{fold_no}.png")
        plot_label_distribution(y_val_fold, f'y_val (Fold {fold_no})', f"results/fold/y_val_labels_{fold_no}.png")


        # X, y, splits = get_classification_data('LSST', split_data=False)
        tfms = [None, TSClassification()]
        splits = (list(train_index), list(val_index))
        batch_tfms = TSStandardize(by_sample=True)
        mv_clf = TSClassifier(X_train, y_train_binary, splits=splits, path='/home/work/.LVEF/ecg-lvef-prediction/models', arch="InceptionTime", tfms=tfms, metrics=accuracy)
        mv_clf.fit_one_cycle(200, 1e-2)
        mv_clf.export("mv_clf.pkl")


        mv_clf = load_learner("models/mv_clf.pkl")
        probas, target, preds = mv_clf.get_X_preds(X_val_fold, y_val_fold)
        print(preds)

        # # 모델 초기화 및 학습
        # splits = (list(range(len(X_train_fold))), list(range(len(X_train_fold), len(X_train_fold) + len(X_val_fold))))
        # dls = get_ts_dls(np.concatenate((X_train_fold, X_val_fold), axis=0),
        #                  np.concatenate((y_train_fold, y_val_fold), axis=0),
        #                  splits=splits,
        #                  batch_size=batch_size)
        
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = TSSequencerPlus(dls.vars, 2, seq_len=1000).to(device)

        # # 모델 구조 요약 출력
        # print("Model Summary:")
        # summary(model, input_size=(4, 1000))

        # learn = Learner(dls, model, metrics=accuracy, loss_func=loss_func)
        # learn.fit_one_cycle(n_epochs)  
        # # 모델 리스트에 추가
        # models.append(learn)

        # ROC 곡선 및 평가 메트릭 저장
        for X, y, dataset in [(X_int, y_int_binary, "int"), (X_ext, y_ext_binary, "ext")]:
            probas, target, preds = mv_clf.get_X_preds(X, y)
            y_pred = preds.astype(int)
            print(y_pred)
            print(y)
            plot_confusion_matrix(
                y,
                y_pred,
                class_names,
                f'Confusion Matrix - {dataset} (Fold {fold_no})',
                f"results/fold_{fold_no}/{dataset}_CM.png"
            )

        fold_no += 1


# Confusion Matrix 시각화 함수
def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    # 저장 경로의 디렉토리를 생성 (존재하지 않으면 생성)
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


if __name__ == "__main__":
    train_cls_with_ensemble()
