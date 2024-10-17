import pandas as pd
import numpy as np
import pickle
import neurokit2 as nk


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
file_path = '/home/work/.LVEF/ecg-lvef-prediction/LBBB_Echo_sorted.csv'

# 데이터 불러오기
df = pd.read_csv(file_path)

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
    sample_data = df.iloc[i, channel_start:channel_end].values.reshape(num_channels, sample_size)
    
    # I, II 채널 선택
    lead_I = sample_data[0]
    lead_II = sample_data[1]

    
    # III, aVR, aVL, aVF 채널 계산
    lead_III = lead_II - lead_I
    aVR = - (lead_I + lead_II) / 2
    aVL = (lead_I - lead_II) / 2
    aVF = (lead_I + lead_II) / 2
    
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
    ef_value = df.iloc[i, 1]  # EF 값 추출
    y_final_data.append(ef_value)
    
    print(f'Sample {i+1}/{num_samples} completed')


# 최종 결과를 DataFrame으로 변환
final_data = np.array(samples)  # (1149, 12, 5000) 형태의 배열

# 데이터 전체에 4.88을 곱한 후 1000으로 나누어 μV에서 mV로 변환
final_data = (final_data * amplitude_units_per_bit) / 1000  # μV -> mV 변환
final_data = final_data.transpose(0, 2, 1)

# 데이터 변환 실행
final_data = transform_data(final_data)

from scipy import signal

# 5000, 4 데이터를 1000, 4 데이터로 리샘플링하는 함수
def resample_data(data, target_size=1000):
    resampled_data = []
    
    for sample in data:
        # 각 채널을 리샘플링하여 새로운 배열에 저장
        resampled_sample = signal.resample(sample, target_size, axis=0)  # axis=0은 시간 축에 대해 리샘플링을 수행
        resampled_data.append(resampled_sample)
    
    return np.array(resampled_data)

# 리샘플링 실행
final_data = resample_data(final_data, target_size=1000)

# 리샘플링 후 결과 확인
print(f'resampeld_data shape: {final_data.shape}')  # (1149, 1000, 4)




# EF 값을 100으로 나누어 정규화
y_final_data = np.array(y_final_data) / 100

# 결과 확인
print(final_data.shape)  
print(final_data)
print(y_final_data.shape)  # (1149,)

# 데이터 분할
train_samples = final_data[:712]  # 첫 번째 712개 샘플 (Train)
train_labels = y_final_data[:712]

int_samples = final_data[712:890]  # 다음 178개 샘플 (Internal Test)
int_labels = y_final_data[712:890]

ext_samples = final_data[890:]  # 나머지 259개 샘플 (External Test)
ext_labels = y_final_data[890:]

# 데이터 저장
data = {
    "train": {"x": train_samples, "y": train_labels},
    "int test": {"x": int_samples, "y": int_labels},
    "ext test": {"x": ext_samples, "y": ext_labels},
}


import matplotlib.pyplot as plt

# 샘플링 레이트 및 시각화 시간 설정
sampling_rate = 100  # Hz
duration = 10  # 초
num_samples = sampling_rate * duration  # 총 샘플 수
time = np.linspace(0, duration, num_samples)  # x축 시간 생성

# 시각화할 샘플 인덱스
sample_index = 11  # 첫 번째 샘플

# 각 채널을 시각화
num_channels = final_data.shape[2]  # 채널 수
fig, axs = plt.subplots(num_channels, 1, figsize=(12, 2 * num_channels), sharex=True)

for channel in range(num_channels):
    axs[channel].plot(time, final_data[sample_index, :, channel], label=f'Sample {sample_index + 1}', color='blue')
    axs[channel].set_title(f'Channel {channel + 1}')
    axs[channel].set_ylabel('Amplitude (mV)')
    axs[channel].grid(True)
    axs[channel].set_ylim(-3, 3)  # 각 서브플롯의 y축 범위 설정


# x축 레이블 설정
axs[-1].set_xlabel('Time (s)')  

plt.tight_layout()
plt.savefig('/home/work/.LVEF/ecg-lvef-prediction/samples.png')
plt.show()



# pickle로 저장
with open("/home/work/.LVEF/ecg-lvef-prediction/data/processed_echo.pkl", "wb") as f:
    pickle.dump(data, f)

print("Data processing and saving completed.")