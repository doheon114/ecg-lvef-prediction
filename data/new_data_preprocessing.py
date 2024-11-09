import pandas as pd
import numpy as np
import pickle
import neurokit2 as nk

# 새로운 (5000, 12) 데이터로 변환하는 함수 정의
def change_data(final_data):
    transformed_samples = []
    
    for sample in final_data:
        # 각 채널의 데이터를 변환하여 새로운 배열에 저장
        new_sample = np.zeros((5000, 12))  # (5000, 12) 형태로 초기화
        
        # 첫 번째 채널 (Lead I)
        new_sample[:, 0] = sample[:, 0]  # I 채널 그대로

        # 두 번째 채널 (Lead II)
        new_sample[:, 1] = sample[:, 1]  # II 채널 그대로

        # 세 번째 채널 (Lead III)
        new_sample[:, 2] = sample[:, 8]  # III 채널 (미리 계산한 값)

        # 네 번째 채널 (aVR)
        new_sample[:, 3] = sample[:, 9]  # aVR 채널 (미리 계산한 값)

        # 다섯 번째 채널 (aVL)
        new_sample[:, 4] = sample[:, 10]  # aVL 채널 (미리 계산한 값)

        # 여섯 번째 채널 (aVF)
        new_sample[:, 5] = sample[:, 11]  # aVF 채널 (미리 계산한 값)

        # V1 ~ V6 채널
        new_sample[:, 6] = sample[:, 2]  # V1
        new_sample[:, 7] = sample[:, 3]  # V2
        new_sample[:, 8] = sample[:, 4]  # V3
        new_sample[:, 9] = sample[:, 5]  # V4
        new_sample[:, 10] = sample[:, 6]  # V5
        new_sample[:, 11] = sample[:, 7]  # V6


        # 변환된 샘플을 리스트에 추가
        transformed_samples.append(new_sample)

    return np.array(transformed_samples)

# 새로운 (5000, 4) 데이터로 변환하는 함수 정의
def transform_data(final_data):
    transformed_samples = []
    
    for sample in final_data:
        # 각 채널의 데이터를 변환하여 새로운 배열에 저장
        new_sample = np.zeros((5000, 4))  # (5000, 4) 형태로 초기화
        
        # 첫 번째 채널 (I, aVR, V1, V4)
        new_sample[:1250, 0] = sample[:1250, 0]  # I의 1~1250 포인트
        new_sample[1250:2500, 0] = sample[1250:2500, 9]  # aVR의 1251~2500 포인트
        new_sample[2500:3750, 0] = sample[2500:3750, 2]  # V1의 2501~3750 포인트
        new_sample[3750:5000, 0] = sample[3750:5000, 5]  # V4의 3751~5000 포인트

        # 두 번째 채널 (II, aVL, V2, V5)
        new_sample[:1250, 1] = sample[:1250, 1]  # II의 1~1250 포인트
        new_sample[1250:2500, 1] = sample[1250:2500, 10]  # aVL의 1251~2500 포인트
        new_sample[2500:3750, 1] = sample[2500:3750, 3]  # V2의 2501~3750 포인트
        new_sample[3750:5000, 1] = sample[3750:5000, 6]  # V5의 3751~5000 포인트

        # 세 번째 채널 (III, aVF, V3, V6)
        new_sample[:1250, 2] = sample[:1250, 8]  # III의 1~1250 포인트
        new_sample[1250:2500, 2] = sample[1250:2500, 11]  # aVF의 1251~2500 포인트
        new_sample[2500:3750, 2] = sample[2500:3750, 4]  # V3의 2501~3750 포인트
        new_sample[3750:5000, 2] = sample[3750:5000, 7]  # V6의 3751~5000 포인트

        # 네 번째 채널 (II 그대로)
        new_sample[:, 3] = sample[:, 1]  # II의 1~5000 포인트 그대로

        # 변환된 샘플을 리스트에 추가
        transformed_samples.append(new_sample)

    return np.array(transformed_samples)


# 파일 경로
file_path = '/home/work/.LVEF/ecg-lvef-prediction/dataset/LBBB_Echo_sorted.csv'

# 데이터 불러오기
data = pd.read_csv(file_path)


# 'id_date' 열을 사용하여 ID와 Date 추출
data['ID'] = data['id_date'].str.split('_').str[0].str.lstrip('0')  
print(data['ID'])
data['Date'] = data['id_date'].str.split('_').str[1]  # 두 번째 부분을 Date로 저장

# Convert 'Date' column to datetime for filtering by date range
data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d%H%M%S')

# Define the list of specific IDs that should always be classified as 'Train'
exception_ids = [
    172732, 178271, 183080, 186047, 200738, 218325, 262914, 309859, 316423, 324953, 
    372932, 405004, 411968, 418886, 433766, 506624, 520902, 542497, 552085, 581120, 
    590721, 626382, 631408, 672894, 686643, 727981, 734265, 805128, 816438, 898733, 
    921360, 923987, 943737, 950259, 1030101, 1039373, 1073601, 1101651, 1177114, 1180634, 
    1183310, 1195121, 1223639, 1260419, 1261498, 1303253, 1315353, 1375528, 1400279, 1471541,
    1502135, 1558767, 1613332, 1658465, 1696792, 1909001, 1919067, 1938087, 1989743, 2029523, 
    2035906, 2049647, 2060098, 2063684, 2097796, 2098843, 2100633, 2123052,
    738769, 1221111, 1305442, 1323289, 1445709
]

# ID를 정수형으로 변환 (혹은 exception_ids를 문자열로 변환)
data['ID'] = data['ID'].astype(int)

# Create a new column for classification, default to NaN
data['Set'] = None

# Apply conditions to classify rows based on the 'Date' and 'hospital' values
data.loc[(data['hospital'] == 'Daejeon') & (data['Date'] <= '2023-12-31'), 'Set'] = 'Train'
data.loc[(data['hospital'] == 'Daejeon') & (data['Date'] >= '2024-01-01'), 'Set'] = 'Int Test'
data.loc[data['hospital'] == 'Sejong', 'Set'] = 'Ext Test'

# Overwrite classifications for rows with IDs in the exception list
data.loc[data['ID'].isin(exception_ids), 'Set'] = 'Train'


# 샘플 수 정의
num_samples = 1149
num_channels = 8  # 기존 채널 수
sample_size = 5000
amplitude_units_per_bit = 4.88  # μV per bit


# 새로운 DataFrame을 위한 리스트 초기화
samples = []
y_final_data = []  # EF 값을 저장할 리스트 초기화

# 각 샘플에 대해 데이터 추출
for i in range(num_samples):
    # 각 채널의 시작 인덱스 계산
    channel_start = 3 
    channel_end = 40003
    
    # 데이터 추출 (채널 개수는 8개)
    sample_data = data.iloc[i, channel_start:channel_end].values.reshape(num_channels, sample_size)
    
    # I, II 채널 선택
    lead_I = sample_data[0]
    lead_II = sample_data[1]

    
    # III, aVR, aVL, aVF 채널 계산
    lead_III = lead_II - lead_I
    aVR = - (lead_I + lead_II) / 2
    aVL = (lead_I - lead_III) / 2   
    aVF = (lead_II + lead_III) / 2
    
    # 기존 데이터에 새로운 채널 추가
    new_sample_data = np.vstack([sample_data, lead_III, aVR, aVL, aVF])
    # 각 채널을 clean하기 전에 dtype을 float으로 변환
    new_sample_data = new_sample_data.astype(float)
    

    # 각 채널에 대해 신호 정리
    for channel in range(new_sample_data.shape[0]):
        new_sample_data[channel] = nk.ecg_clean(new_sample_data[channel], sampling_rate=500, method='neurokit')



    
    # samples 리스트에 추가
    samples.append(new_sample_data)
    
    # 2열(EF 값) 정보 저장
    ef_value = data.iloc[i, 1]  # EF 값 추출
    y_final_data.append(ef_value)
    
    print(f'Sample {i+1}/{num_samples} completed')


# 최종 결과를 DataFrame으로 변환
final_data = np.array(samples)  # (1149, 12, 5000) 형태의 배열
print(final_data.shape)


# 데이터 전체에 4.88을 곱한 후 1000으로 나누어 μV에서 mV로 변환
final_data = (final_data * amplitude_units_per_bit) / 1000  # μV -> mV 변환
final_data = final_data.transpose(0, 2, 1)
# final_data = change_data(final_data)

# 데이터 변환 실행
final_data = transform_data(final_data)

from scipy import signal

# 5000, 4 데이터를 1000, 4 데이터로 리샘플링하는 함수
def resample_data(data, target_size=2500):
    resampled_data = []
    
    for sample in data:
        # 각 채널을 리샘플링하여 새로운 배열에 저장
        resampled_sample = signal.resample(sample, target_size, axis=0)  # axis=0은 시간 축에 대해 리샘플링을 수행
        resampled_data.append(resampled_sample)
    
    return np.array(resampled_data)

# 리샘플링 실행
# final_data = resample_data(final_data, target_size=1000)
# 1000, 12 데이터를 500, 12로 나누는 함수 정의
def split_data(final_data):
    split_samples = []
    
    for sample in final_data:
        # 각 샘플을 반으로 나누어 새로운 샘플 생성
        split_samples.append(sample[:2500])   # 첫 번째 절반
        split_samples.append(sample[2500:])   # 두 번째 절반

    return np.array(split_samples)

# split_data 함수 호출
# final_data = split_data(final_data)

# # 각 y 값도 두 배로 만들어야 하므로 y_final_data를 반복하여 증가시킴
# y_final_data = np.repeat(y_final_data, 2)

final_data = final_data.transpose(0, 2, 1)

num_channels_to_add = 8
# 추가할 채널에 대한 0 배열 생성 (각 샘플에 대해 동일한 크기의 0 배열 생성)
zeros_to_add = np.zeros((final_data.shape[0], num_channels_to_add, final_data.shape[2]))

# 기존 데이터와 0 배열 연결 (샘플별로 연결)
final_data = np.concatenate((final_data, zeros_to_add), axis=1)



# 리샘플링 후 결과 확인
print(f'resampeld_data shape: {final_data.shape}')  # (1149, 1000, 4)




# EF 값을 100으로 나누어 정규화
y_final_data = np.array(y_final_data) / 100


# 결과 확인
print(final_data.shape)  
print(final_data)
print(y_final_data.shape)  # (1149,)



train_samples = final_data[data['Set'] == 'Train']
train_labels = y_final_data[data['Set'] == 'Train']

int_samples = final_data[data['Set'] == 'Int Test']
int_labels = y_final_data[data['Set'] == 'Int Test']

ext_samples = final_data[data['Set'] == 'Ext Test']
ext_labels = y_final_data[data['Set'] == 'Ext Test']

print(train_samples.shape)
print(int_samples.shape)
print(ext_samples.shape)

# 데이터 저장
data = {
    "train": {"x": train_samples, "y": train_labels},
    "int test": {"x": int_samples, "y": int_labels},
    "ext test": {"x": ext_samples, "y": ext_labels},
}


# pickle로 저장
with open("/home/work/.LVEF/ecg-lvef-prediction/data/new_processed_(12,5000).pkl", "wb") as f:
    pickle.dump(data, f)

print("Data processing and saving completed.")


# import matplotlib.pyplot as plt

# # 샘플링 레이트 및 시각화 시간 설정
# sampling_rate = 100  # Hz
# duration = 10  # 초
# num_samples = sampling_rate * duration  # 총 샘플 수
# time = np.linspace(0, duration, num_samples)  # x축 시간 생성

# # 시각화할 샘플 인덱스
# sample_index = 2298 # 첫 번째 샘플
# final_data = final_data.transpose(0, 2, 1)

# for sample_index in range(1000, 1050):
#     # 각 채널을 시각화
#     num_channels = final_data.shape[2]  # 채널 수
#     fig, axs = plt.subplots(num_channels, 1, figsize=(12, 2 * num_channels), sharex=True)

#     for channel in range(num_channels):
#         axs[channel].plot(time, final_data[sample_index, :, channel], label=f'Sample {sample_index + 1}', color='blue')
#         axs[channel].set_title(f'Channel {channel + 1}')
#         axs[channel].set_ylabel('Amplitude (mV)')
#         axs[channel].grid(True)
#         axs[channel].set_ylim(-3, 3)  # 각 서브플롯의 y축 범위 설정


#     # x축 레이블 설정
#     axs[-1].set_xlabel('Time (s)')  

#     plt.tight_layout()
#     plt.savefig(f'/home/work/.LVEF/ecg-lvef-prediction/samples/samples_{sample_index}.png')
#     plt.show()
#     plt.close()


# for sample_index in range(50, 100):
#     # 각 채널을 시각화
#     num_channels = final_data.shape[2]  # 채널 수
#     fig, axs = plt.subplots(num_channels, 1, figsize=(12, 2 * num_channels), sharex=True)

#     for channel in range(num_channels):
#         axs[channel].plot(time, final_data[sample_index, :, channel], label=f'Sample {sample_index + 1}', color='blue')
#         axs[channel].set_title(f'Channel {channel + 1}')
#         axs[channel].set_ylabel('Amplitude (mV)')
#         axs[channel].grid(True)
#         axs[channel].set_ylim(-3, 3)  # 각 서브플롯의 y축 범위 설정


#     # x축 레이블 설정
#     axs[-1].set_xlabel('Time (s)')  

#     plt.tight_layout()
#     plt.savefig(f'/home/work/.LVEF/ecg-lvef-prediction/samples/samples_{sample_index}.png')
#     plt.show()
#     plt.close()

