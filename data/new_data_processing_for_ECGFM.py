import os
import numpy as np
import pickle
from scipy.io import loadmat, savemat
import torch
import wfdb
import pandas as pd

# 1. .mat 파일 불러오기
# file_path = '/home/work/.LVEF/ecg-lvef-prediction/data/new_data/training/sample_1.mat'
# data = loadmat(file_path)

# .mat 파일의 구조를 확인합니다
# print(data.keys())

# 2. 데이터 불러오기: 파일의 특정 키에 데이터가 있을 수 있으므로, 이를 확인 후 수정해야 합니다
# 예를 들어, 실제 데이터가 'data' 키에 있다면 다음과 같이 할 수 있습니다:
# ecg_data = data['idx']  # 실제 키 이름으로 변경 필요
# 데이터가 (샘플 수, 채널 수, 타임포인트 수) 형태인지 확인합니다.
# print(ecg_data)

# import matplotlib.pyplot as plt
# import numpy as np

# # 예시 ECG 데이터 (12, 2500)
# # ecg_data = np.random.randn(12, 2500)  # 실제 데이터를 여기에 할당

# def plot_ecg_data(ecg_data, save_path="ecg_plot.png"):
#     num_leads, num_samples = ecg_data.shape
#     time = np.linspace(0, num_samples / 250, num_samples)  # 샘플링 주파수에 따라 시간 축 설정 (250Hz 기준)

#     # 플롯 설정
#     fig, axes = plt.subplots(num_leads, 1, figsize=(10, 12), sharex=True)
#     fig.suptitle("12-lead ECG Data Visualization", fontsize=16)

#     for i in range(num_leads):
#         axes[i].plot(time, ecg_data[i], label=f"Lead {i+1}")
#         axes[i].set_ylabel(f"Lead {i+1}")
#         axes[i].legend(loc="upper right")
    
#     axes[-1].set_xlabel("Time (s)")
#     plt.tight_layout(rect=[0, 0, 1, 0.96])  # 상단 타이틀과의 간격 조절
#     plt.savefig(save_path, dpi=300)  # 이미지 파일로 저장
#     plt.close(fig)  # 메모리 절약을 위해 플롯 닫기

# # 코드 실행 예시
# plot_ecg_data(ecg_data, save_path="/home/work/.LVEF/ecg-lvef-prediction/ecg_plot_my_org.png")

with open("/home/work/.LVEF/ecg-lvef-prediction/data/final_combined_data.pkl", "rb") as f:
    data = pickle.load(f)

# 데이터 PyTorch 텐서로 변환
X_train = torch.tensor(data["X_train"], dtype=torch.float32)
y_train = torch.tensor(data["y_train"], dtype=torch.float32)
X_int = torch.tensor(data["X_int"], dtype=torch.float32)
y_int = torch.tensor(data["y_int"], dtype=torch.float32)
X_ext = torch.tensor(data["X_ext"], dtype=torch.float32)
y_ext = torch.tensor(data["y_ext"], dtype=torch.float32)
X_ext_two = torch.tensor(data["X_ext_two"], dtype=torch.float32)
y_ext_two = torch.tensor(data["y_ext_two"], dtype=torch.float32)



output_dir = '/home/work/.LVEF/ecg-lvef-prediction/data/new_data/training'  # 원하는 저장 경로로 변경
# 변환 함수
def convert_to_binary(y):
    return np.column_stack([np.where(y < 0.4, 0, 1), np.where(y < 0.4, 1, 0)])

# 변환된 배열
y_train_binary = convert_to_binary(y_train)
y_int_binary = convert_to_binary(y_int)
y_ext_binary = convert_to_binary(y_ext)
y_ext_two_binary = convert_to_binary(y_ext_two)

# 세 배열을 쌓기
y_combined = np.vstack([y_train_binary, y_int_binary, y_ext_binary, y_ext_two_binary])

# .npy 파일로 저장
np.save('/home/work/.LVEF/ecg-lvef-prediction/y.npy', y_combined)

print("저장 완료: y_combined.npy")
# 각 샘플을 개별 .mat 파일로 저장
# for i, sample in enumerate(X_ext_two):
#     # 각 채널의 평균과 표준편차 계산    
#     mean_values = torch.mean(sample, dim=1).reshape(-1, 1).numpy()  # 각 채널의 평균을 열 형태로 변환
#     std_values = torch.std(sample, dim=1).reshape(-1, 1).numpy()    # 각 채널의 표준편차를 열 형태로 변환
    
    
#     # 각 샘플을 .mat 형식으로 저장
#     sample_path = os.path.join(output_dir, f'sample_{i+1810}.mat')
#     savemat(sample_path, {
#         'org_sample_rate': 500,
#         'curr_sample_rate': 500,
#         'org_sample_size': 5000,
#         'curr_sample_size': 5000,
#         'feats': sample,
#         'idx': i+1809,
#         'mean': mean_values,
#         'std': std_values,
#         'segment_i': 0
#     })

#     # 저장 완료 확인
#     print(f'Saved sample {i+1} to {sample_path}')


# import os
# import wfdb

# from scipy.io import loadmat

# # .mat 파일 경로 설정
# output_hea_dir = "/home/work/.LVEF/ecg-lvef-prediction/data/new_data/training/Int"  # .hea 파일을 저장할 경로

# # y_ext 값 (예시로 임의의 값을 사용. 실제 값에 맞게 설정)

# # 리드 이름 정의 (순서에 맞춰 설정)
# lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# # 샘플 ID를 순차적으로 처리
# for idx, y_value in enumerate(y_int):
#     sample_id = f"sample_{idx+1}"  # 예: sample_1, sample_2, ...
    
#     # .mat 파일 경로 설정 (샘플 ID에 맞는 .mat 파일 경로)
#     mat_file_path = f"/home/work/.LVEF/ecg-lvef-prediction/data/new_data/training/Int/{sample_id}.mat"
    
#     # 리드에 맞는 진단 값 설정
#     dx_value = 1 if y_value < 0.4 else 0  # y_value에 따라 Dx 값 설정 (예: y_value < 0.4 이면 1, 아니면 0)
    
#     # .mat 파일에서 리드 정보와 관련된 데이터를 읽음
#     try:
#         # .mat 파일 로드
#         mat_data = loadmat(mat_file_path)
        
#         # 'val' 키에서 ECG 데이터 추출
#         feats = mat_data['val']  # 'val' 키에 ECG 데이터가 저장되어 있다고 가정
        
#         # 각 리드에 대한 정보 (12개의 리드) 추출
#         leads_info = []
#         for i, lead_name in enumerate(lead_names):  # 리드 순서에 맞게 순차적으로 처리
#             first_value = feats[0, i]  # 첫 번째 샘플의 값
#             checksum = 0  # 체크섬 값은 0으로 가정 (필요에 따라 변경 가능)
#             leads_info.append((lead_name, first_value, checksum))

#         # 샘플 ID에 맞는 .hea 파일 경로 설정
#         hea_file_path = os.path.join(output_hea_dir, f"{sample_id}.hea")

#         # .hea 파일 작성
#         with open(hea_file_path, "w") as f:
#             # 첫 번째 라인: 샘플 ID, 리드 개수, 샘플링 주파수, 샘플 수
#             f.write(f"{sample_id} 12 500 5000\n")
            
            
            
#             # 각 리드에 대한 정보를 파일에 작성
#             for lead, first_value, checksum in leads_info:
#                 f.write(f"{sample_id}.mat 16x1+24 1000.0(0)/mV 16 0 {first_value} {checksum} 0 {lead}\n")

#             # 진단 정보 작성
#             f.write(f"# Dx: {dx_value}\n")

#         print(f"{hea_file_path} 생성 완료 - Dx: {dx_value}")
    
#     except Exception as e:
#         print(f"파일 {mat_file_path}를 처리하는 중 오류 발생: {e}")

# print("y_train에 대한 모든 샘플의 .hea 파일 생성 완료.")

# import os
# import pandas as pd

# # 기본 경로와 설정
# base_dir = "/home/work/.LVEF/ecg-lvef-prediction/data/new_data/training"
# csv_output_path = "/home/work/.LVEF/ecg-lvef-prediction/my_own_final/physionet2021/records.csv"  # 저장할 CSV 파일 경로

# # 처리할 폴더 리스트
# folders = ["Ext", "Int", "Train&val"]

# # 파일 리스트 생성
# data = []
# for folder in folders:
#     folder_path = os.path.join(base_dir, folder)
#     for file_name in os.listdir(folder_path):
#         if file_name.endswith('.mat'):
#             # '폴더명/파일명' 형식으로 기록, 확장자 제거
#             file_path = os.path.join(folder, file_name.split('.')[0])
#             data.append([file_path, folder])

# # DataFrame으로 변환 후 CSV 저장 (기존 파일을 덮어쓰거나 새로 생성)
# df = pd.DataFrame(data, columns=['path', 'dataset'])
# df.to_csv(csv_output_path, index=False)

# print(f"CSV 파일이 {csv_output_path}에 저장되었습니다.")

