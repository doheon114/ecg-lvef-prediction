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

use_residual = False
use_bottleneck = False
depth = 6
kernel_size = 20
n_filters = 32
batch_size = 16

# InceptionTimeClassifier 모델 초기화 및 로드
clf = InceptionTimeClassifier(verbose=True,
                              kernel_size=kernel_size,
                              n_filters=n_filters,
                              use_residual=use_residual,
                              use_bottleneck=use_bottleneck,
                              depth=depth,
                              random_state=0).build_model(input_shape=(1000, 4), n_classes=2)

model_path = "/home/work/.LVEF/ecg-lvef-prediction/results/cls/InceptionTimeClassifier_(1000, 4)/fold_2/model_weights.h5"
clf.load_weights(model_path)

# class0과 class1 데이터를 각각 5개씩 추출
class0_indices = np.where(y_train < 0.4)[0][:5]
class1_indices = np.where(y_train >= 0.4)[0][:5]

# # 클래스 0에 대한 LIME 시각화
for i, id_ecg in enumerate(class0_indices):
    instance_ecg = X_train[id_ecg, :]
    print(f"Class 0 - Example {i+1}: Shape of instance_ecg:", instance_ecg.shape)

    # 예측
    probability_vector = clf.predict(instance_ecg[np.newaxis, :])
    top_pred_classes, predicted_class = analyze_prediction(probability_vector, [0, 1])

    # Segmentation
    num_slices = 100
    slice_width = segment_ecg_signal(instance_ecg, num_slices)

    # plot_segmented_ecg(instance_ecg, slice_width, f"/home/work/.LVEF/ecg-lvef-prediction/ecg_visualizations_real/LVSD_example{i+1}_plot_segment.png")

    # Perturbation
    num_perturbations = 350
    random_perturbations = generate_random_perturbations(num_perturbations, num_slices)
    perturbed_ecg_example = apply_perturbation_to_ecg(instance_ecg, random_perturbations[-1], num_slices, perturb_mean)

    # plot the original and perturbed ECG signals with highlighted slices and deactivated segments
    # plot_perturbed_ecg(instance_ecg, perturbed_ecg_example, random_perturbations[-1], num_slices, f"/home/work/.LVEF/ecg-lvef-prediction/ecg_visualizations_real/LVSD_example{i+1}_plot_perturbed.png", title='ECG Signal with Perturbation')


    # Perturbation Predictions
    perturbation_predictions = predict_perturbations(clf, instance_ecg, random_perturbations, num_slices, perturb_mean)

    # Calculate distances and weights
    cosine_distances = calculate_cosine_distances(random_perturbations, num_slices)
    weights = calculate_weights_from_distances(cosine_distances, kernel_width=0.25)

    # Explainable Model
    segment_importance_coefficients = fit_explainable_model(perturbation_predictions, random_perturbations, weights, target_class=top_pred_classes[0])

    # Top influential segments
    top_influential_segments = identify_top_influential_segments(segment_importance_coefficients, number_of_top_features=5)


    # 시각화
    visualize_lime_explanation(instance_ecg, top_influential_segments, num_slices, f"/home/work/.LVEF/ecg-lvef-prediction/ecg_visualizations_real/LVSD_example{i+1}.png", perturb_function=perturb_mean)

# 클래스 1에 대한 LIME 시각화
for i, id_ecg in enumerate(class1_indices):
    instance_ecg = X_train[id_ecg, :]
    print(f"Class 1 - Example {i+1}: Shape of instance_ecg:", instance_ecg.shape)

    # 예측
    probability_vector = clf.predict(instance_ecg[np.newaxis, :])
    top_pred_classes, predicted_class = analyze_prediction(probability_vector, [0, 1])

    # Segmentation
    num_slices = 100
    slice_width = segment_ecg_signal(instance_ecg, num_slices)

    # Perturbation
    num_perturbations = 350
    random_perturbations = generate_random_perturbations(num_perturbations, num_slices)
    perturbed_ecg_example = apply_perturbation_to_ecg(instance_ecg, random_perturbations[-1], num_slices, perturb_mean)

    # Perturbation Predictions
    perturbation_predictions = predict_perturbations(clf, instance_ecg, random_perturbations, num_slices, perturb_mean)

    # Calculate distances and weights
    cosine_distances = calculate_cosine_distances(random_perturbations, num_slices)
    weights = calculate_weights_from_distances(cosine_distances, kernel_width=0.25)

    # Explainable Model
    segment_importance_coefficients = fit_explainable_model(perturbation_predictions, random_perturbations, weights, target_class=top_pred_classes[0])

    # Top influential segments
    top_influential_segments = identify_top_influential_segments(segment_importance_coefficients, number_of_top_features=5)

    # 시각화
    visualize_lime_explanation(instance_ecg, top_influential_segments, num_slices, f"/home/work/.LVEF/ecg-lvef-prediction/ecg_visualizations_real/class1_example{i+1}.png", perturb_function=perturb_mean)
