from sktime.regression.deep_learning import InceptionTimeRegressor
from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor
from sklearn.metrics import PredictionErrorDisplay, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import pickle
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 랜덤 시드 고정
random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
random.seed(random_seed)

# 데이터 로드
with open("data/processed.pkl", "rb") as f:
    data = pickle.load(f)
    X_train = data["train"]["x"]
    y_train = data["train"]["y"]
    X_int = data["int test"]["x"]
    y_int = data["int test"]["y"]
    X_ext = data["ext test"]["x"]
    y_ext = data["ext test"]["y"]

# 5겹 교차 검증 설정
kf = KFold(n_splits=5, shuffle=False)

mae_scores = []
r2_scores = []

for n_epochs in [100]:
    fold = 1
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        # InceptionTimeRegressor 모델 초기화 및 훈련
        clf = InceptionTimeRegressor(n_epochs=n_epochs, verbose=True, metrics=[tf.keras.metrics.MeanAbsoluteError()], random_state=random_seed).build_model(input_shape=(1000, 4))
        
        clf.summary()
        clf.fit(X_train_fold, y_train_fold, batch_size=128, validation_data=(X_val_fold, y_val_fold), epochs=n_epochs)

        # 내부 및 외부 테스트 셋에 대해 예측 수행 및 결과 시각화
        for X, y, dataset in [(X_int, y_int, "int"), (X_ext, y_ext, "ext")]:
            y_pred = clf.predict(X)
            y_pred = y_pred.reshape(-1)  # (189, 1)을 (189,)으로 변환

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
            plt.savefig(f"results/{dataset}_E{n_epochs}_Fold{fold}.png")
            plt.close(fig)

        fold += 1

# MAE와 R² 점수의 평균 계산
mean_mae = np.mean(mae_scores)
mean_r2 = np.mean(r2_scores)

print(f"Average MAE: {mean_mae:.3f}")
print(f"Average R² Score: {mean_r2:.3f}")
