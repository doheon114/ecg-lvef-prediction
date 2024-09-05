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
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, precision_recall_curve
from sklearn.model_selection import KFold
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize, label_binarize
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

# Custom functions
from utils_3 import evaluate_metrics_multiclass, vis_history

# Hyperparameters
th = [0.4, 0.5]  # Thresholds
n_epochs = 10
x_shape = (1000, 4)
k_folds = 5

use_residual = False
use_bottleneck = False
depth = 6
kernel_size = 20
n_filters = 32
batch_size = 16

# Data Loading
with open("data/processed.pkl", "rb") as f:
    data = pickle.load(f)
    num = x_shape[1]
    X_train = data["train"]["x"]
    y_train = data["train"]["y"]
    X_int = data["int test"]["x"]
    y_int = data["int test"]["y"]
    X_ext = data["ext test"]["x"]
    y_ext = data["ext test"]["y"]

# Label Binarization and Multi-Class Conversion
def binarize_labels(y, thresholds):
    return np.digitize(y, bins=thresholds, right=True)

y_train_multi = binarize_labels(y_train, th)
y_int_multi = binarize_labels(y_int, th)
y_ext_multi = binarize_labels(y_ext, th)

# Label Distribution Visualization
def plot_label_distribution(y_data, dataset_name, save_path):
    label_counts = dict(Counter(y_data))
    plt.figure(figsize=(6, 4))
    bars = plt.bar(label_counts.keys(), label_counts.values(), color=['blue', 'orange', 'green'])
    plt.xticks(ticks=range(len(label_counts)), labels=[f'Class {i}' for i in label_counts.keys()])
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title(f'Label Distribution in {dataset_name}')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), ha='center', va='bottom')
    plt.savefig(save_path)
    plt.close()

# Plot label distribution for each dataset
plot_label_distribution(y_train_multi, 'y_train', "/home/work/.LVEF/ecg-lvef-prediction/results/y_train_labels.png")
plot_label_distribution(y_int_multi, 'y_int', "/home/work/.LVEF/ecg-lvef-prediction/results/y_int_labels.png")
plot_label_distribution(y_ext_multi, 'y_ext', "/home/work/.LVEF/ecg-lvef-prediction/results/y_ext_labels.png")

# ROC and Precision-Recall Curve Calculation
def compute_roc_prc(y_true, y_pred_proba, n_classes):
    roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
    prc_auc = average_precision_score(y_true, y_pred_proba, average='weighted')

    # Compute ROC curve and PR curve
    fpr, tpr, _ = roc_curve(y_true.ravel(), y_pred_proba.ravel())
    precision, recall, _ = precision_recall_curve(y_true.ravel(), y_pred_proba.ravel())
    
    return roc_auc, prc_auc, fpr, tpr, precision, recall

# Plot Combined ROC and Precision-Recall Curves
def plot_combined_roc_curves(roc_data, prc_data, PATH):
    plt.figure(figsize=(16, 6))
    
    # ROC Curve Plot
    plt.subplot(1, 2, 1)
    for dataset, (roc_auc, fpr, tpr) in roc_data.items():
        plt.plot(fpr, tpr, lw=2, label=f'{dataset} (AUROC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate [%]')
    plt.ylabel('True Positive Rate [%]')
    plt.title('ROC Curve for Multi-Class Classification')
    plt.legend(loc='lower right')
    
    # Precision-Recall Curve Plot
    plt.subplot(1, 2, 2)
    for dataset, (prc_auc, precision, recall) in prc_data.items():
        plt.plot(recall, precision, lw=2, label=f'{dataset} (AUPRC = {prc_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall [%]')
    plt.ylabel('Precision [%]')
    plt.title('Precision-Recall Curve for Multi-Class Classification')
    plt.legend(loc='lower left')
    
    plt.savefig(PATH + "combined_curves.png")
    plt.show()

# K-Fold Training and Evaluation
def train_cls(base):
    times = datetime.today().strftime("%Y%m%d_%H:%M:%S")
    class_names = ["EF<40%", "40%≤EF<50%", "EF≥50%"]

    int_accuracies = []
    ext_accuracies = []

    for lr in [0.000005]:
        PATH = f"results/cls/{times}_{base}_clip0.3/{lr}/"
        os.makedirs(PATH, exist_ok=True)

        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=0)
        fold_no = 1

        for train_index, val_index in kfold.split(X_train):
            print(f"Training on fold {fold_no}...")
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train_multi[train_index], y_train_multi[val_index]

            # Validation set label distribution visualization
            plot_label_distribution(y_train_fold, f'y_train (Fold {fold_no})', f"/home/work/.LVEF/ecg-lvef-prediction/results/fold/y_train_labels_{fold_no}.png")
            plot_label_distribution(y_val_fold, f'y_val (Fold {fold_no})', f"/home/work/.LVEF/ecg-lvef-prediction/results/fold/y_val_labels_{fold_no}.png")

            # Compute class weights
            class_weight = compute_class_weight('balanced', classes=np.unique(y_train_fold), y=y_train_fold)
            class_weight_dict = dict(enumerate(class_weight))
            print(f"Fold {fold_no} - Class Weights: {class_weight_dict}")

            # Model creation and compilation
            clf = InceptionTimeClassifier(
                verbose=True,
                kernel_size=kernel_size,
                n_filters=n_filters,
                use_residual=use_residual,
                use_bottleneck=use_bottleneck,
                depth=depth,
                random_state=0
            ).build_model(input_shape=x_shape, n_classes=3)

            clf.compile(optimizer=Adam(learning_rate=lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            clf.summary()

            # Training the model with class weights
            history = clf.fit(
                X_train_fold, y_train_fold,
                validation_data=(X_val_fold, y_val_fold),
                epochs=n_epochs,
                batch_size=batch_size,
                shuffle=True,
                class_weight=class_weight_dict
            )

            vis_history(history, PATH + f"fold_{fold_no}/", lr)

            # ROC and Precision-Recall Curve Data
            roc_data = {}
            prc_data = {}

            for X, y_multi, dataset in [(X_int, y_int_multi, "Internal"), (X_ext, y_ext_multi, "External")]:
                y_pred_proba = clf.predict(X)
                y_true_bin = label_binarize(y_multi, classes=np.arange(3))

                # Confusion Matrix
                ConfusionMatrixDisplay.from_predictions(
                    y_multi,
                    np.argmax(y_pred_proba, axis=1),
                    display_labels=class_names,
                    cmap=plt.cm.Blues,
                )
                plt.savefig(PATH + f"fold_{fold_no}/{dataset}_CM.png")
                plt.close()

                # ROC and Precision-Recall Data Calculation
                roc_auc, prc_auc, fpr, tpr, precision, recall = compute_roc_prc(y_true_bin, y_pred_proba, n_classes=3)

                if dataset == "Internal":
                    roc_data['Internal'] = (roc_auc, fpr, tpr)
                    prc_data['Internal'] = (prc_auc, precision, recall)
                elif dataset == "External":
                    roc_data['External'] = (roc_auc, fpr, tpr)
                    prc_data['External'] = (prc_auc, precision, recall)

                # Record evaluation metrics in history.json
                metrics = evaluate_metrics_multiclass(y_multi, np.argmax(y_pred_proba, axis=1), y_pred_proba)
                print(f"Dataset: {dataset}, Fold: {fold_no}, Learning Rate: {lr}")
                print(f"Accuracy: {metrics['accuracy']}")
                print(f"Sensitivity (Recall): {metrics['sensitivity']}")
                print(f"Specificity: {metrics['specificity']}")
                print(f"F1-score: {metrics['f1_score']}")
                print(f"AUROC: {metrics['auroc']}")
                print(f"AUPRC: {metrics['auprc']}")

                with open(PATH + f"fold_{fold_no}/history.json", "a") as f:
                    json.dump({dataset: metrics}, f, indent=4)

                if dataset == "Internal":
                    int_accuracies.append(metrics['accuracy'])
                elif dataset == "External":
                    ext_accuracies.append(metrics['accuracy'])

            # Combined ROC curves visualization
            plot_combined_roc_curves(roc_data, prc_data, PATH + f"fold_{fold_no}/")

            fold_no += 1

    # Mean accuracies across folds
    print(f"\nMean accuracy for Internal dataset across {k_folds} folds: {np.mean(int_accuracies):.4f}")
    print(f"Mean accuracy for External dataset across {k_folds} folds: {np.mean(ext_accuracies):.4f}")

if __name__ == "__main__":
    train_cls(f"{x_shape[0], x_shape[1]}_{th}")
