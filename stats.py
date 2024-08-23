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
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf

print("TensorFlow version:", tf.__version__)


with open("/home/work/.LVEF/ecg-lvef-prediction/data/processed.pkl", "rb") as f:
    data = pickle.load(f)
    X_train = data["train"]["x"]
    y_train = data["train"]["y"]
    X_int = data["int test"]["x"]
    y_int = data["int test"]["y"]
    X_ext = data["ext test"]["x"]
    y_ext = data["ext test"]["y"]

# 샘플 데이터 생성 (여기서는 랜덤 데이터로 예를 듭니다)
np.random.seed(0)
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# ADF 테스트 함수 정의
def adf_test(series):
    result = adfuller(series)
    return result[0], result[1], result[4]

# 결과를 저장할 리스트
results = []

# 모든 샘플과 채널에 대해 ADF 테스트 수행
for sample in range(X_train.shape[0]):
    for channel in range(X_train.shape[2]):  # 채널 수는 X_train.shape[2]로 동적으로 설정
        time_series = X_train[sample, :, channel]  # 올바른 인덱스 사용
        adf_stat, p_value, critical_values = adf_test(time_series)
        results.append({
            'Sample': sample,
            'Channel': channel,
            'ADF Statistic': adf_stat,
            'p-value': p_value,
            'Critical Values': critical_values
        })

# 결과를 DataFrame으로 변환
results_df = pd.DataFrame(results)

# 결과 출력
print(results_df.head())

# 조건에 따라 결과를 저장할 수 있습니다.
# results_df.to_csv('adf_test_results.csv', index=False)
