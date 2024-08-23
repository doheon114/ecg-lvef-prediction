from sktime.regression.deep_learning import InceptionTimeRegressor
from sktime.classification.deep_learning import InceptionTimeClassifier
from sklearn.metrics import PredictionErrorDisplay, ConfusionMatrixDisplay, r2_score
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import os
import datetime
import tensorflow as tf
import random
from sklearn.model_selection import TimeSeriesSplit


th = 0.5

# 시드 설정
def set_seeds(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

print("TensorFlow version:", tf.__version__)


with open("/home/work/.LVEF/ecg-lvef-prediction/data/processed.pkl", "rb") as f:
    data = pickle.load(f)
    X_train = data["train"]["x"]
    y_train = data["train"]["y"]
    X_int = data["int test"]["x"]
    y_int = data["int test"]["y"]
    X_ext = data["ext test"]["x"]
    y_ext = data["ext test"]["y"]


def train_reg():
    # 시드 설정
    set_seeds(seed=0)

    log_dir_base = "/home/work/.LVEF/ecg-lvef-prediction/work/logs/"
    checkpoint_dir = "/home/work/.LVEF/ecg-lvef-prediction/work/checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 모델을 초기화하고 초기 가중치를 저장합니다.
    clf = InceptionTimeRegressor(n_filters=24, kernel_size=100,  use_residual=True, use_bottleneck=False, verbose=True, random_state=0).build_model(input_shape=(250, 12))
    clf.compile(optimizer=Adam(learning_rate=0.0005), loss="mean_squared_error", metrics=["mean_absolute_error"])

    #initial_weights = clf.get_weights()  # 초기 가중치 저장

    tscv = TimeSeriesSplit(n_splits=5)  # 교차 검증의 fold 수를 설정합니다.

    for fold, (train_index, val_index) in enumerate(tscv.split(X_train)):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # 각 폴드 전에 모델을 초기 가중치로 재설정합니다.
        #clf.set_weights(initial_weights)

        # TensorBoard와 ModelCheckpoint 콜백 설정
        log_dir = log_dir_base + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"_fold_{fold}"
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        #checkpoint_path = os.path.join(checkpoint_dir, f"best_model_fold_{fold}.h5")
        #checkpoint_callback = ModelCheckpoint(
        #    filepath=checkpoint_path,
        #    save_best_only=True,
        #    monitor='val_loss',
        #    mode='min',
        #    verbose=1
        #)

        clf.summary()

        history = clf.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold),
                          epochs=50, callbacks=[tensorboard_callback],
                          batch_size=128, shuffle=False)

        # 최적의 모델 로드
        #clf.load_weights(checkpoint_path)

        for X, y, dataset in [(X_int, y_int, "int"), (X_ext, y_ext, "ext")]:
            y_pred = np.squeeze(clf.predict(X))

            r2 = r2_score(y, y_pred)
            mae = np.mean(np.abs(y - y_pred))

            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))

            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=np.array(y_pred),
                kind="actual_vs_predicted",
                ax=axs[0],
            )
            axs[0].set_title("Actual vs. Predicted values")
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=np.array(y_pred),
                kind="residual_vs_predicted",
                ax=axs[1],
            )
            axs[1].set_title("Residuals vs. Predicted Values")
            axs[0].set_xlim(0, 0.75)  # X축 범위 설정
            axs[0].set_ylim(0, 0.75)  # Y축 범위 설정
            axs[0].set_xticks(np.arange(0, 0.71, 0.1))
            axs[0].set_yticks(np.arange(0, 0.71, 0.1))
            axs[1].set_xticks(np.arange(0, 0.71, 0.1))

            fig.suptitle(f"Fold {fold} - MAE={mae:.3f}, R2={r2:.3f}")
            plt.tight_layout()
            plt.savefig(f"/home/work/.LVEF/ecg-lvef-prediction/results/reg/{dataset}_fold_{fold}.png")
 
if __name__ == "__main__":
    train_reg()
