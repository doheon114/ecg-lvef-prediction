from sktime.regression.deep_learning import InceptionTimeRegressor
from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor
from sklearn.metrics import PredictionErrorDisplay
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import MeanAbsoluteError

with open ("data/processed.pkl", "rb") as f:
    data = pickle.load(f)
    X_train = data["train"]["x"]
    y_train = data["train"]["y"]
    X_int = data["int test"]["x"]
    y_int = data["int test"]["y"]
    X_ext = data["ext test"]["x"]
    y_ext = data["ext test"]["y"]

for n_epochs in [10, 50, 100, 200, 500, 1000]:
    clf = InceptionTimeRegressor(n_epochs=n_epochs, verbose=True, metrics=[MeanAbsoluteError], random_state=0)
    clf.fit(X_train, y_train)

    for X, y, dataset in [(X_int, y_int, "int"), (X_ext, y_ext, "ext")]:
        y_pred = clf.predict(X)

        fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
        PredictionErrorDisplay.from_predictions(
            y,
            y_pred=y_pred,
            kind="actual_vs_predicted",
            ax=axs[0],
        )
        axs[0].set_title("Actual vs. Predicted values")
        PredictionErrorDisplay.from_predictions(
            y,
            y_pred=y_pred,
            kind="residual_vs_predicted",
            ax=axs[1],
        )
        axs[1].set_title("Residuals vs. Predicted Values")
        axs[0].set_xticks(np.arange(0, 0.71, 0.1))
        axs[0].set_yticks(np.arange(0, 0.71, 0.1))
        axs[1].set_xticks(np.arange(0, 0.71, 0.1))

        fig.suptitle(f"Plotting cross-validated predictions (MAE={np.mean(np.abs(y - y_pred)):.3f})")
        plt.tight_layout()
        plt.savefig(f"results/{dataset}_E{n_epochs}.png")