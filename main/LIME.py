from tensorflow.keras.models import load_model
from sktime.classification.deep_learning import InceptionTimeClassifier
import numpy as np
import pickle
from lime_explanation import analyze_prediction
from lime_explanation import segment_ecg_signal, generate_random_perturbations
from visualization import plot_segmented_ecg
from lime_explanation import apply_perturbation_to_ecg, perturb_mean
from visualization import plot_perturbed_ecg
from lime_explanation import predict_perturbations
from lime_explanation import calculate_cosine_distances
from lime_explanation import calculate_weights_from_distances
from lime_explanation import fit_explainable_model
from lime_explanation import identify_top_influential_segments
from visualization import visualize_lime_explanation





x_shape = (1000, 4)

# 데이터 로드
with open("data/processed.pkl", "rb") as f:
    data = pickle.load(f)
    num = x_shape[1]
    X_train = data["train"]["x"]
    y_train = data["train"]["y"]
    X_int = data["int test"]["x"]
    y_int = data["int test"]["y"]
    X_ext = data["ext test"]["x"]
    y_ext = data["ext test"]["y"]

use_residual=False
use_bottleneck=False
depth=6
kernel_size=20
n_filters=32 
batch_size=16

clf = InceptionTimeClassifier(verbose=True,
                                        kernel_size=kernel_size, 
                                        n_filters=n_filters, 
                                        use_residual=use_residual,
                                        use_bottleneck=use_bottleneck, 
                                        depth=depth, 
                                        random_state=0).build_model(input_shape=(1000, 4), n_classes=2)
            
# 모델 로드
model_path = "/home/work/.LVEF/ecg-lvef-prediction/results/cls/InceptionTimeClassifier_best/fold_4/model_weights.h5"
clf.load_weights(model_path)

id_ecg = 0
instance_ecg = X_train[id_ecg, :]
print("Shape of instance_ecg:", instance_ecg.shape)

# 예측
probability_vector = clf.predict(instance_ecg[np.newaxis, :])
print("Predicted probability vector:", probability_vector)

# class labels
class_labels = [0, 1]


top_pred_classes, predicted_class = analyze_prediction(probability_vector, class_labels)

print("Top predicted classes:", top_pred_classes)
print("Predicted Class for the selected instance:", predicted_class)

# Segmentation using the fixed number of slices
num_slices = 100
slice_width = segment_ecg_signal(instance_ecg, num_slices)

# plot the segmented ECG signal
plot_segmented_ecg(instance_ecg, slice_width, "/home/work/.LVEF/ecg-lvef-prediction/ecg_visualizations_real/segmented.png")

# Perturbation
num_perturbations = 150
random_perturbations = generate_random_perturbations(num_perturbations, num_slices)

# Example output
print("The shape of random_perturbations array (num_perturbations, num_slices):", random_perturbations.shape)
print("Example Perturbation:", random_perturbations[-1])

# Choose the perturbation function
perturb_function = perturb_mean  

# Apply a random perturbation to the ECG signal
perturbed_ecg_example = apply_perturbation_to_ecg(instance_ecg, random_perturbations[-1], num_slices, perturb_function)

# plot the original and perturbed ECG signals with highlighted slices and deactivated segments
plot_perturbed_ecg(instance_ecg, perturbed_ecg_example, random_perturbations[-1], num_slices, "/home/work/.LVEF/ecg-lvef-prediction/ecg_visualizations_real/perturbed.png", title='ECG Signal with Perturbation')


## Predict the class probabilities using the trained ECG classifier
perturbation_predictions = predict_perturbations(clf, instance_ecg, random_perturbations, num_slices, perturb_mean)


# Calculate cosine distances between each perturbation and the original ECG signal representation
cosine_distances = calculate_cosine_distances(random_perturbations, num_slices)
print("Shape of Cosine Distances Array:", cosine_distances.shape)


#Applying a Kernel Function to Compute Weights
kernel_width = 0.25  # This can be adjusted based on your specific needs
weights = calculate_weights_from_distances(cosine_distances, kernel_width)

# Now we have the weights for each perturbation for further analysis
print("Shape of Weights Array:", weights.shape)

# Check the shape of perturbation predictions
print("Shape of perturbation_predictions:", perturbation_predictions.shape)


# Constructing the Explainable Model for ECG Signals
segment_importance_coefficients = fit_explainable_model(perturbation_predictions, random_perturbations, weights, target_class=top_pred_classes[0])

# The importance coefficients for each segment
print("Segment Importance Coefficients:", segment_importance_coefficients)


number_of_top_features = 5
top_influential_segments = identify_top_influential_segments(segment_importance_coefficients, number_of_top_features)

# The indices of the top influential segments
print("Top Influential Signal Segments:", top_influential_segments)

visualize_lime_explanation(instance_ecg, top_influential_segments, num_slices, "/home/work/.LVEF/ecg-lvef-prediction/ecg_visualizations_real/final.png",perturb_function=perturb_mean)