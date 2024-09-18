import tensorflow as tf
import os
import json
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from datetime import datetime
from utils_2 import evaluate_metrics, vis_history
import numpy as np
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries

# 경로 설정
visualization_path = "/home/work/.LVEF/ecg-lvef-prediction/ecg_visualizations"

# 하이퍼파라미터 설정
th = 0.4  # 기준값 설정
n_epochs = 2
batch_size = 64
input_shape = (1600, 512, 3)  # 이미지의 크기 및 채널 수 설정 (2 채널 이미지)

# ResNet50 모델 정의
def create_resnet50(input_shape, n_classes):
    base_model = ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')  # ResNet50 모델 사용
    base_model.trainable = False  # Pre-trained 모델 고정

    x = layers.GlobalAveragePooling2D()(base_model.output)
    output = layers.Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

# 데이터셋 로드 함수
def load_image_data(data_dir, img_size=(1600, 512), batch_size=16):
    dataset = image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",  # 이진 분류에 사용
        shuffle=True
    )
    return dataset

# LIME 설명 시각화 함수
def plot_lime_explanation(img_array, model):
    # 모델 예측 함수 정의
    def predict_fn(images):
        images = np.array([img for img in images])  # 이미지를 배열로 변환
        preds = model.predict(images)
        return preds

    # LIME 이미지 해석기 생성
    explainer = lime_image.LimeImageExplainer()

    # 첫 번째 이미지 선택
    img = img_array[0].numpy()  # 이미지를 numpy 형식으로 변환

    # LIME을 이용하여 설명 생성
    explanation = explainer.explain_instance(
        img.astype('double'),
        predict_fn,  # 모델 예측 함수
        top_labels=2,  # 상위 2개의 레이블에 대한 설명 생성
        hide_color=0,
        num_samples=1000  # 샘플링 수
    )

    # 선택된 레이블에 대한 설명 시각화 (첫 번째 클래스)
    temp, mask = explanation.get_image_and_mask(
        label=np.argmax(model.predict(img_array)[0]),  # 예측된 클래스의 설명
        positive_only=False,
        num_features=10,
        hide_rest=False
    )

    # LIME 마스크를 이미지 위에 시각화
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(img.astype("uint8"))
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(mark_boundaries(temp, mask))
    plt.title("LIME Explanation")
    plt.show()

# 데이터셋 로드
X_train_path = os.path.join(visualization_path, 'X_train')
X_int_path = os.path.join(visualization_path, 'X_int')
X_ext_path = os.path.join(visualization_path, 'X_ext')

train_ds = load_image_data(X_train_path)
int_ds = load_image_data(X_int_path)
ext_ds = load_image_data(X_ext_path)

# Data splitting function for TensorFlow Dataset
def split_dataset(dataset, split_ratio=0.8):
    dataset_size = len(dataset)
    split_index = int(dataset_size * split_ratio)
    train_dataset = dataset.take(split_index)
    val_dataset = dataset.skip(split_index)
    return train_dataset, val_dataset

# 모델 학습 함수
def train_resnet_with_ensemble():
    times = datetime.today().strftime("%Y%m%d_%H:%M:%S")
    class_names = ["EF<40%", "EF>40%"]

    # 모델들을 저장할 리스트
    models = []

    for lr in [0.0005]:
        PATH = f"results/cls/{times}_resnet50/"
        os.makedirs(PATH, exist_ok=True)

        # 데이터셋을 train과 validation으로 나눔
        train_split_ds, val_ds = split_dataset(train_ds)

        # 모델 학습 및 평가
        model = create_resnet50(input_shape=input_shape, n_classes=2)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        model.summary()

        history = model.fit(train_split_ds, validation_data=val_ds, epochs=n_epochs, batch_size=batch_size)
        vis_history(history, PATH, lr)

        # 모델 저장
        models.append(model)

        # ROC 및 PRC 곡선 계산 및 저장

        # LIME을 이용한 시각화 (일부 이미지에 대해 수행)
        for images, labels in int_ds.take(1):  # 임의의 배치에서 첫 번째 이미지 사용
            img_array = images[0:1]  # 첫 번째 이미지 추출
            plot_lime_explanation(img_array, model)

# ROC 및 PRC 곡선 플로팅 함수
def plot_combined_roc_curves(roc_data, prc_data, save_path):
    plt.figure(figsize=(16, 6))

    # ROC 곡선
    plt.subplot(1, 2, 1)
    for dataset, (fpr, tpr, roc_auc) in roc_data.items():
        plt.plot(fpr, tpr, lw=2, label=f'{dataset} (AUROC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')

    # Precision-Recall 곡선
    plt.subplot(1, 2, 2)
    for dataset, (precision, recall, prc_auc) in prc_data.items():
        plt.plot(recall, precision, lw=2, label=f'{dataset} (AUPRC = {prc_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')

    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    train_resnet_with_ensemble()
