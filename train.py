# import 

import json
import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf

from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError as MSELoss
from tensorflow.keras.metrics import BinaryCrossentropy, AUC, Accuracy, TruePositives, FalsePositives, TrueNegatives, FalseNegatives
from tensorflow.keras.losses import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

# custom
#from src.visualization import show_history 


class Trainer :
    def __init__(self, model, epochs, types, idx, save_path) :
        self.model = model
        self.epochs = epochs
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.types = types
        self.idx = idx
        self.save_path = save_path

        # callbacks
        # self.loss_cb = ModelCheckpoint(self.save_path / f"{self.idx}_model", monitor="val_loss", 
        #                           save_best_only=True, mode="min", verbose=1)
        # self.acc_cb = ModelCheckpoint(self.save_path / f"{self.idx}_model", monitor="val_auc", 
        #                          save_best_only=True, mode="max", verbose=1)
        self.loss_fn = None
        self.METRICS = None    
        print(self.types)
        if "cls" in self.types :
            self.loss_fn = BinaryCrossentropy()
            self.METRICS = [
                        TruePositives(name="tp"),
                        FalsePositives(name="fp"),
                        TrueNegatives(name="tn"),
                        FalseNegatives(name="fn"),
                        Accuracy(name="acc"),
                        AUC(name="auc"),
                        AUC(name="prc", curve="PR"),
                        ]
            
        elif self.types == "reg":
            self.loss_fn = MSELoss()
            self.METRICS = [
                        MeanSquaredError(name="mse"),
                        MeanAbsoluteError(name="mae"),
                        ]
   

    def train(self, X_train, y_train, X_val, y_val):
        
        # 모델 컴파일
        self.model.compile(optimizer=self.optimizer, 
                        loss=self.loss_fn, 
                        metrics=self.METRICS)
        
        # 모델 학습
        history = self.model.fit(X_train, np.array(y_train), validation_data=(X_val, np.array(y_val)), epochs=self.epochs, verbose=1)

        # 학습 이력 저장
        with open("/Users/doheonkim/Desktop/history.json", "w") as f:
            json.dump(history.history, f)
        
        # 검증 데이터에 대한 예측 수행
        y_pred = self.model.predict(X_val)

        # Bland-Altman 플롯 생성
        plot_bland_altman(y_val, y_pred)
        
        # 산점도 플롯 생성
        plot_scatter(y_val, y_pred)
    
    def test(self, model, test_dl, METRICS) :
        #for cb in [self.loss_cb, self.acc_cb] :
        #   model = load_model(cb)
        model.compile(optimizer=self.optimizer, 
                   loss=self.loss_fn, 
                    metrics=METRICS)
        model.evaluate(test_dl)



def plot_scatter(y_true, y_pred):
    # 산점도 플롯

    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, color='blue', s=10)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Scatter Plot of True vs Predicted Values')
    plt.show()

def plot_bland_altman(y_true, y_pred):
    # 평균과 차이 계산
    print(y_pred)

    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)


    mean = np.mean([y_true, y_pred], axis=0)
    diff = y_true - y_pred
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

    # Bland-Altman 플롯
    plt.figure(figsize=(10, 6))
    plt.scatter(mean, diff, color='blue', s=10)
    plt.axhline(mean_diff, color='red', linestyle='--')
    plt.axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle='--')
    plt.axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle='--')
    plt.xlabel('Mean of True and Predicted Values')
    plt.ylabel('Difference between True and Predicted Values')
    plt.title('Bland-Altman Plot')
    plt.show()
        
        


        #show_history(history, self.save_path / f"{self.idx}.png")

        