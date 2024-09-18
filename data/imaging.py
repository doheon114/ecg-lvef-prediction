import pickle
import matplotlib.pyplot as plt
import numpy as np
import os 

# 데이터 로드
with open("data/processed.pkl", "rb") as f:
    data = pickle.load(f)
    X_train = data["train"]["x"]
    X_int = data["int test"]["x"]
    X_ext = data["ext test"]["x"]
    y_train = data["train"]["y"]
    y_int = data["int test"]["y"]
    y_ext = data["ext test"]["y"]  # 대응되는 y 값 로드

visualization_path = "/home/work/.LVEF/ecg-lvef-prediction/ecg_visualizations_real/"

def visualize_ecg(ecg_data, filepath, offset=5, dpi=100):
    plt.figure(figsize=(16, 9), dpi=dpi)  # 픽셀 크기 1600x512, DPI=100 설정
    time = np.linspace(0, 10, ecg_data.shape[0])  # Assuming the longest strip is 10 seconds

    # Create a figure and set its background color
    fig = plt.gcf()
    fig.patch.set_facecolor('lightgrey')  # Set figure background color

    ax = plt.gca()  # Get current axes
    ax.set_facecolor('lightgrey')  # Set axes background color

    # 각 리드를 서로 떨어져 표시하기 위해 오프셋을 더해줌
    for i in range(ecg_data.shape[1]):
        plt.plot(time, ecg_data[:, i] + i * offset, color='black', linewidth=1)  # 오프셋 추가
    
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # 그리드 설정

    os.makedirs(os.path.dirname(filepath), exist_ok=True)  # 디렉토리 없으면 생성
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0.1)  # 여백 조정
    plt.close()

# Loop through each sample in X_ext and save based on y value
for idx, sample in enumerate(X_train):
    if y_train[idx] < 0.4:
        folder = "class_0"
    else:
        folder = "class_1"
    
    # 저장할 경로 설정
    filename = f"train_sample_{idx}.png"
    filepath = os.path.join(visualization_path, folder, filename)
    
    # 시각화 및 저장
    visualize_ecg(sample, filepath)
