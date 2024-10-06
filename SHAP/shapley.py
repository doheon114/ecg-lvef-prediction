import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle
import tensorflow as tf

from tensorflow.keras import layers, optimizers, losses, metrics, activations, regularizers, callbacks
from keras.models import Model
from sklearn.preprocessing import LabelBinarizer
from sktime.classification.deep_learning import InceptionTimeClassifier

# ------------------------------
# 1. Setup and Data Loading
# ------------------------------

# Set GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)  # Memory growth must be set at program startup

# Parameters
use_residual = False
use_bottleneck = False
depth = 6
kernel_size = 20
n_filters = 32 
batch_size = 16

x_shape = (1000, 4)

# Load data
with open("data/processed.pkl", "rb") as f:
    data = pickle.load(f)
    X_train = data["train"]["x"]
    y_train = data["train"]["y"]
    X_int = data["int test"]["x"]
    y_int = data["int test"]["y"]
    X_ext = data["ext test"]["x"]
    y_ext = data["ext test"]["y"]

# Thresholding
th = 0.4
y_train = (y_train >= th).astype(np.int64)
y_int = (y_int >= th).astype(np.int64)
y_ext = (y_ext >= th).astype(np.int64)

# One-Hot Encoding
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_int = lb.transform(y_int)
y_ext = lb.transform(y_ext)

print("Classes:", lb.classes_)

# ------------------------------
# 2. Model Loading
# ------------------------------

# Initialize and build the model
model = InceptionTimeClassifier(
    verbose=True,
    kernel_size=kernel_size, 
    n_filters=n_filters, 
    use_residual=use_residual,
    use_bottleneck=use_bottleneck, 
    depth=depth, 
    random_state=0,
).build_model(input_shape=(1000, 4), n_classes=2)

# Load pre-trained weights
model.load_weights("/home/work/.LVEF/ecg-lvef-prediction/results/cls/InceptionTimeClassifier_(1000, 4)/fold_2/model_weights.h5")

# ------------------------------
# 3. SHAP Value Calculation
# ------------------------------

# Select first 10 samples from internal test set
X_train_100 = X_train[:100]  # Shape: (10, 1000, 4)
X_int_50 = X_int[:50]

print("X_train_100 shape:", X_train_100.shape)

# Initialize DeepExplainer
explainer = shap.DeepExplainer(model, X_train_100)
print("Explainer initialized.")

# Compute SHAP values
shap_values = explainer.shap_values(X_int_50) 
print("SHAP values computed. Shape:", np.array(shap_values).shape)

# ------------------------------
# 4. Visualization
# ------------------------------


# Parameters for visualization
sample_number = range(0, 50)  # Index of the sample to visualize
for i in sample_number:
    sample_idx = i
    target_class_idx = 0  # Index of the target class

    # Extract ECG data for the selected sample
    ecg_data = X_int_50[sample_idx]  # Shape: [1000, 4]
    signal_length, num_leads = ecg_data.shape

    # Extract SHAP values for the target class
    if isinstance(shap_values, list):
        target_shap_values = shap_values[target_class_idx][sample_idx]  # Shape: [1000, 4]
    else:
        target_shap_values = shap_values[sample_idx, :, :, target_class_idx]

    # Check if the predicted class for the sample is 0
    predicted_class = np.argmax(model.predict(np.expand_dims(ecg_data, axis=0)), axis=1)[0]
    if predicted_class == 0:
        # Identify the top N SHAP values
        N = 200 # Number of top SHAP values to plot
        flat_indices = np.argsort(np.abs(target_shap_values).ravel())[-N:]
        time_indices, lead_indices = np.unravel_index(flat_indices, target_shap_values.shape)

        # Create a time array
        sampling_rate = 100  # Hz
        time = np.linspace(0, signal_length / sampling_rate, signal_length)

        # Initialize the plot
        fig, axes = plt.subplots(num_leads, 1, figsize=(15, 10), sharex=True)

        # Determine global SHAP scaling factor to maintain alignment across leads
        shap_scale_factor = (np.max(ecg_data) - np.min(ecg_data)) * 0.1  # 10% of ECG range

        # Plot each lead with SHAP overlay
        for lead_idx in range(num_leads):
            lead_data = ecg_data[:, lead_idx]  # Shape: [1000]
            axes[lead_idx].plot(time, lead_data, lw=1, color='black')
                       # Set y-axis limits to [-3, 3] for all plots
            axes[lead_idx].set_ylim([-3, 3])

            # Find top SHAP indices for this lead
            current_lead_mask = lead_indices == lead_idx
            current_time_indices = time_indices[current_lead_mask]
            current_shap_values = target_shap_values[current_time_indices, lead_idx]

            # Scale SHAP values
            scaled_shap_values = current_shap_values * shap_scale_factor

            # Overlay the SHAP values on the ECG plot
            shap_overlay = lead_data[current_time_indices] + scaled_shap_values
            axes[lead_idx].scatter(time[current_time_indices], shap_overlay, color='orange', s=20, alpha=0.7)

        # Final plot adjustments
        axes[-1].set_xlabel('Time (seconds)')
        plt.suptitle(f'ECG Signals with Top 200 SHAP Values (LVSD)', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save and display the plot
        plt.savefig(f"/home/work/.LVEF/ecg-lvef-prediction/SHAP/LVSD_{i+1}_top200features.png")
        plt.show()