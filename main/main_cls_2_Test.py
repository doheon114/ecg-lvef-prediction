from sktime.classification.deep_learning import InceptionTimeClassifier, CNNClassifier, ResNetClassifier, MLPClassifier, LSTMFCNClassifier, CNTCClassifier
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
import dask.dataframe as dd

print("hello")
warnings.filterwarnings("ignore")

# custom
from utils_2 import evaluate_metrics, vis_history







# hyperparameters
th = 0.4  # 기준값을 0.5로 설정
n_epochs = 200
x_shape = (1000,4)  
k_folds = 5

use_residual=False
use_bottleneck=False
depth=6
kernel_size=20
n_filters=32 
batch_size=16
# data
with open("/home/work/.LVEF/ecg-lvef-prediction/data/processed.pkl", "rb") as f:
    data = pickle.load(f)
    num = x_shape[1]
    X_train = data["train"]["x"]
    y_train = data["train"]["y"]
    X_int = data["int test"]["x"]
    y_int = data["int test"]["y"]
    X_ext = data["ext test"]["x"]
    y_ext = data["ext test"]["y"]
# data
with open("/home/work/.LVEF/ecg-lvef-prediction/data/processed_echo.pkl", "rb") as f:
    data = pickle.load(f)
    num = x_shape[1]
    X_train_new = data["train"]["x"]
    y_train_new = data["train"]["y"]
    X_int_new = data["int test"]["x"]
    y_int_new = data["int test"]["y"]
    X_ext_new = data["ext test"]["x"]
    y_ext_new = data["ext test"]["y"]

import numpy as np

# 데이터 합치기 (X_train, X_train_new 등)
X_train = X_train
y_train = y_train

X_int = np.concatenate((X_int, X_train_new, X_int_new), axis=0)
y_int = np.concatenate((y_int, y_train_new, y_int_new), axis=0)

X_ext = X_ext
y_ext = y_ext
X_ext_new = X_ext_new
y_ext_new = y_ext_new

# 합친 데이터의 shape 확인
print("X_train_combined shape:", X_train.shape)
print("y_train_combined shape:", y_train.shape)

print("X_int_combined shape:", X_int.shape)
print("y_int_combined shape:", y_int.shape)

print("X_ext shape:", X_ext.shape)
print("y_ext shape:", y_ext.shape)

print("X_ext_new shape:", X_ext_new.shape)
print("y_ext_new shape:", y_ext_new.shape)




# X_train = X_train.reshape(X_train.shape[0], 4, 1000)  # X_train의 shape을 (n_samples, 4, 1000)으로 변환
# X_int = X_int.reshape(X_int.shape[0], 4, 1000)         # X_int의 shape을 (n_samples, 4, 1000)으로 변환
# X_ext = X_ext.reshape(X_ext.shape[0], 4, 1000)         # X_ext의 shape을 (n_samples, 4, 1000)으로 변환


# 레이블을 이진화한 후의 데이터셋들
y_train_binary = (y_train >= th).astype(np.int64)
y_int_binary = (y_int >= th).astype(np.int64)
y_ext_binary = (y_ext >= th).astype(np.int64)
y_ext_new_binary = (y_ext_new >= th).astype(np.int64)


# y 라벨 비율 계산 및 시각화 함수 정의
def plot_label_distribution(y_data, dataset_name, save_path):
    label_counts = dict(Counter(y_data))
    plt.figure(figsize=(6, 4))
    bars = plt.bar(label_counts.keys(), label_counts.values(), color=['blue', 'orange'])
    plt.xticks(ticks=[0, 1], labels=["LVSD", "No LVSDS"])
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

def ensemble_predict(models, X_data, voting='soft'):
    """
    앙상블 예측 함수: 하드 보팅 또는 소프트 보팅을 사용해 다수 모델의 예측을 결합
    Args:
        models (list): 훈련된 모델들의 리스트
        X_data (np.array): 예측할 데이터셋
        voting (str): 'hard' 또는 'soft' 보팅 방식 선택
    Returns:
        np.array: 앙상블된 최종 예측 결과
    """
    if voting == 'soft':
        # 소프트 보팅 - 각 모델의 예측 확률을 평균
        y_preds = np.array([model.predict(X_data) for model in models])
        y_prob_avg = np.mean(y_preds, axis=0)  # 각 클래스에 대한 확률의 평균
        y_pred = np.argmax(y_prob_avg, axis=1)  # 평균 확률에 따라 최종 클래스 예측
    else:
        # 하드 보팅 - 각 모델의 예측 결과(클래스)를 다수결로 결정
        y_preds = np.array([np.argmax(model.predict(X_data), axis=1) for model in models])
        y_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=y_preds)

    return y_pred

def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """
    Confusion Matrix를 시각화하고 저장하는 함수
    Args:
        y_true (np.array): 실제 라벨
        y_pred (np.array): 예측 라벨
        class_names (list): 클래스 이름 리스트
        title (str): 시각화 제목
        save_path (str): 저장 경로
    """
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

    int_accuracies = []
    ext_accuracies = []

    # 모델들을 저장할 리스트
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


            plot_label_distribution(y_train_fold, f'y_train (Fold {fold_no})', f"/home/work/.LVEF/ecg-lvef-prediction/results/fold/y_train_labels_{fold_no}.png")
            plot_label_distribution(y_val_fold, f'y_val (Fold {fold_no})', f"/home/work/.LVEF/ecg-lvef-prediction/results/fold/y_val_labels_{fold_no}.png")

            class_weight = compute_class_weight('balanced', classes=np.unique(y_train_fold), y=y_train_fold)
            class_weight_dict = dict(enumerate(class_weight))

            print(f"Fold {fold_no} - Class Weights: {class_weight_dict}")

            clf = InceptionTimeClassifier(verbose=True,
                                        kernel_size=kernel_size, 
                                        n_filters=n_filters, 
                                        use_residual=use_residual,
                                        use_bottleneck=use_bottleneck, 
                                        depth=depth, 
                                        random_state=0,
                                        ).build_model(input_shape=(1000, 4), n_classes=2)
           
            clf.compile(optimizer=Adam(learning_rate=lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            clf.summary()

            # history = clf.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold), epochs=n_epochs, batch_size=batch_size, class_weight=class_weight_dict)
            clf.load_weights("/home/work/.LVEF/ecg-lvef-prediction/results/cls/InceptionTimeClassifier_(1000, 4)/fold_2/model_weights.h5")
            # vis_history(history, PATH + f"fold_{fold_no}/", lr)

            # 모델을 리스트에 추가
            models.append(clf)

            # ROC 곡선 및 평가 메트릭 저장
            roc_data = {}
            prc_data = {}

            for X, y, dataset in [(X_int, y_int_binary, "int"), (X_ext, y_ext_binary, "ext"),  (X_ext_new, y_ext_new_binary, "ext_new")]:
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
                # if dataset == "int":
                #     int_accuracies.append(metrics['accuracy'])
                # elif dataset == "ext":
                #     ext_accuracies.append(metrics['accuracy'])

            # Combined ROC curves 시각화
            plot_combined_roc_curves(roc_data, prc_data, PATH + f"fold_{fold_no}/")


            fold_no += 1

    # 훈련된 모델들을 사용하여 앙상블 예측 수행
    for dataset, X, y in [("int", X_int, y_int_binary), ("ext", X_ext, y_ext_binary), ("ext_new", X_ext_new, y_ext_new_binary)]:
        # 소프트 보팅
        y_pred_soft = ensemble_predict(models, X, voting='soft')
        # 하드 보팅
        y_pred_hard = ensemble_predict(models, X, voting='hard')

        # 평가 및 Confusion Matrix 시각화
        print(f"\n=== {dataset.upper()} DATASET ENSEMBLE RESULTS ===")
        for voting_type, y_pred in zip(['soft', 'hard'], [y_pred_soft, y_pred_hard]):
            print(f"\nEnsemble Voting: {voting_type.capitalize()}")
            metrics = evaluate_metrics(y, y_pred, y_pred)  # 평가 메트릭 계산
            print(f"Accuracy: {metrics['accuracy']}")
            print(f"Sensitivity (Recall): {metrics['sensitivity']}")
            print(f"Specificity: {metrics['specificity']}")
            print(f"F1-score: {metrics['f1_score']}")
            print(f"AUROC: {metrics['auroc']}")
            print(f"AUPRC: {metrics['auprc']}")

            # 평가 메트릭을 history.json 파일에 기록
            with open(PATH + f"{voting_type}_history.json", "a") as f:
                json.dump({dataset: metrics}, f, indent=4)

            # Confusion Matrix 시각화 및 저장
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