# Ejection Fraction Prediction from 12-Lead ECG

## Overview

This repository contains code for predicting the ejection fraction (EF) of the left ventricle using 12-lead electrocardiogram (ECG) data. The prediction is based on time series regression techniques, specifically utilizing the `KNeighborsTimeSeriesRegressor` from the `sktime` library.

## Code Description

1. **Data Loading:**
   - The ECG data is stored in a pickled file named `processed.pkl`.
   - The data is split into three sets: training, internal testing (int), and external testing (ext).
   - Features (X) and target labels (y) are extracted for each set.

2. **Model Training:**
   - A `KNeighborsTimeSeriesRegressor` model is instantiated.
   - The model is trained on the training data (X_train, y_train).

3. **Prediction and Visualization:**
   - For both the internal and external test sets:
     - Predictions are made using the trained model (y_pred).
     - Prediction errors are visualized using two plots:
       - Actual vs. Predicted values
       - Residuals vs. Predicted values
     - The resulting plots are saved as `int.png` and `ext.png`.

## Requirements

- Python 3.x
- Required Python packages: `sktime`, `scikit-learn`, and `matplotlib`

## Usage

1. Ensure you have the necessary Python packages installed.
2. Place your processed ECG data in the `data/processed.pkl` file.
3. Run the provided code to train the model and generate prediction error plots.

## Citation

If you find this code useful, please consider citing the relevant libraries and tools used:

- `sktime`: [GitHub Repository](https://github.com/alan-turing-institute/sktime)
- `scikit-learn`: [Website](https://scikit-learn.org/stable/index.html)