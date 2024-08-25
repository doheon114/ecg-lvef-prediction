from sktime.regression.deep_learning import InceptionTimeRegressor
from sklearn.metrics import PredictionErrorDisplay, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import pickle
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import os



# 랜덤 시드 고정
random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
random.seed(random_seed)

# 텐서플로우에서의 추가적인 재현성 설정
os.environ['PYTHONHASHSEED'] = str(random_seed)  # Python 해시 시드 고정
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

# 데이터 로드
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

# 5겹 교차 검증 설정
kf = KFold(n_splits=5, shuffle=False)

mae_scores = []
r2_scores = []

bottleneck=True
depth=6
kernel_size=20
n_filters=16
batch_size=32

# TensorBoard 로그 디렉토리 설정
log_dir = "/home/work/.LVEF/ecg-lvef-prediction/work/logs/fit"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

for n_epochs in [100]:  
    fold = 1
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        # InceptionTimeRegressor 모델 초기화 및 훈련
        clf = InceptionTimeRegressor(n_epochs=n_epochs, depth=depth, kernel_size=kernel_size, use_bottleneck=bottleneck, n_filters=n_filters, verbose=True, metrics=[tf.keras.metrics.MeanAbsoluteError()], random_state=random_seed)
        model = clf.build_model(input_shape=(1000, 4))
        model.compile(
            optimizer=Adam(learning_rate=0.0005), 
            loss="mean_squared_error", 
            metrics=["mean_absolute_error"]
        )
        model.summary()
        
        # TensorBoard 콜백 설정
        tensorboard_callback = TensorBoard(log_dir=os.path.join(log_dir, f"{bottleneck}_{depth}_{kernel_size}_{n_filters}_{batch_size}/fold_{fold}"), histogram_freq=1)
        
        # 모델 훈련
        model.fit(
            X_train_fold,
            y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=n_epochs,
            batch_size=batch_size,
            callbacks=[tensorboard_callback]
        )

        # 내부 및 외부 테스트 셋에 대해 예측 수행 및 결과 시각화
        for X, y, dataset in [(X_int, y_int, "int"), (X_ext, y_ext, "ext")]:
            y_pred = model.predict(X)
            y_pred = y_pred.reshape(-1) 

            # MAE와 R² 점수 계산
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            mae_scores.append(mae)
            r2_scores.append(r2)

            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=y_pred,
                kind="actual_vs_predicted",
                ax=axs[0],
            )
            axs[0].set_title(f"Actual vs. Predicted values - Fold {fold}")
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=y_pred,
                kind="residual_vs_predicted",
                ax=axs[1],
            )
            axs[1].set_title(f"Residuals vs. Predicted Values - Fold {fold}")
            axs[0].set_xticks(np.arange(0, 0.71, 0.1))
            axs[0].set_yticks(np.arange(0, 0.71, 0.1))
            axs[1].set_xticks(np.arange(0, 0.71, 0.1))

            fig.suptitle(f"Plotting cross-validated predictions (MAE={mae:.3f}, R²={r2:.3f})")
            plt.tight_layout()
            plt.savefig(f"results/reg/{dataset}_E{n_epochs}_Fold{fold}.png")
            plt.close(fig)

        fold += 1

# MAE와 R² 점수의 평균 계산
mean_mae = np.mean(mae_scores)
mean_r2 = np.mean(r2_scores)

print(f"Average MAE: {mean_mae:.3f}")
print(f"Average R² Score: {mean_r2:.3f}")
