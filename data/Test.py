import os
import numpy as np
import matplotlib.pyplot as plt

# XML 파일에서 ECG 데이터를 로드하는 함수
def XMLloader(filename):
    with open(f'{filename}', 'r', encoding='utf-8') as f:
        ecg = f.read()  # 파일 내용을 읽어옴
    tmp = ecg.split(">")

    # ECG 데이터가 담긴 딕셔너리 생성
    ecg_dict = {
        "I": np.array(list(map(float, tmp[4][:-10].split(" ")))),
        "aVR": np.array(list(map(float, tmp[16][:-12].split(" ")))),
        "V1": np.array(list(map(float, tmp[28][:-11].split(" ")))),
        "V4": np.array(list(map(float, tmp[40][:-11].split(" ")))),
        "II": np.array(list(map(float, tmp[8][:-11].split(" ")))),
        "aVL": np.array(list(map(float, tmp[20][:-12].split(" ")))),
        "V2": np.array(list(map(float, tmp[32][:-11].split(" ")))),
        "V5": np.array(list(map(float, tmp[44][:-11].split(" ")))),
        "III": np.array(list(map(float, tmp[12][:-12].split(" ")))),
        "aVF": np.array(list(map(float, tmp[24][:-12].split(" ")))),
        "V3": np.array(list(map(float, tmp[36][:-11].split(" ")))),
        "V6": np.array(list(map(float, tmp[48][:-11].split(" ")))),
        "Rhythm strip": np.array(list(map(float, tmp[52][:-17].split(" "))))
    }
    return ecg_dict

# 리드를 동일한 길이로 변환하는 함수
def interpolate_signals(signals):
    interpolated_signals = {}
    for lead, signal in signals.items():
        if lead == "Rhythm strip":
            target_length = 1000
        else:
            target_length = 250
        
        x = np.linspace(0, len(signal), target_length, endpoint=False)
        xp = np.linspace(0, len(signal), len(signal), endpoint=False)
        interpolated_signals[lead] = np.interp(x, xp, signal)
    return interpolated_signals

# 신호를 -1에서 1로 정규화하는 함수
def normalize_signal(signal):
    min_val = np.min(signal)
    max_val = np.max(signal)
    return 2 * (signal - min_val) / (max_val - min_val) - 1

# 각 리드를 y축 평행 이동시키는 함수
def align_leads(ecg_signals):
    aligned_signals = {}
    prev_end = 0
    for lead in ecg_signals.keys():
        current_signal = ecg_signals[lead]
        current_end = current_signal[-1]
        offset = prev_end - current_signal[0]
        aligned_signals[lead] = current_signal + offset
        prev_end = aligned_signals[lead][-1]
    return aligned_signals

# 개별 XML 파일 처리 및 시각화 함수
def process_and_visualize_ecg(filename, save_path):
    # 샘플링 주파수 설정 (Hz)
    sampling_rate = 250  # 250 Hz

    # 각 샘플링 포인트 간의 시간 간격 계산 (밀리초 단위)
    time_interval = 1000 / sampling_rate  # 밀리초 단위로 변환

    # ECG 데이터 로드
    ecg_data = XMLloader(filename)

    # 모든 리드를 동일한 길이로 변환
    ecg_resampled = interpolate_signals(ecg_data)

    # 모든 리드를 -1에서 1로 정규화
    ecg_normalized = {lead: normalize_signal(signal) for lead, signal in ecg_resampled.items()}

    # 모든 리드를 정렬
    ecg_aligned = align_leads(ecg_normalized)

    # 모든 리드를 결합
    ecg_combined = np.stack([
        np.concatenate((ecg_aligned["I"], ecg_aligned["aVR"], ecg_aligned["V1"], ecg_aligned["V4"])),
        np.concatenate((ecg_aligned["II"], ecg_aligned["aVL"], ecg_aligned["V2"], ecg_aligned["V5"])),
        np.concatenate((ecg_aligned["III"], ecg_aligned["aVF"], ecg_aligned["V3"], ecg_aligned["V6"])),
        ecg_aligned["Rhythm strip"]
    ])

    # 시간 벡터를 밀리초 단위로 설정
    time_vector = np.arange(ecg_combined.shape[1]) * time_interval

    # 각 리드를 시각화
    plt.figure(figsize=(12, 10))

    for i, lead in enumerate(ecg_combined):
        plt.subplot(len(ecg_combined), 1, i + 1)
        plt.plot(time_vector, lead)
        plt.title(f'Lead {i + 1}')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 메인 함수
def main():
    xml_folder = "/home/work/.LVEF/ecg-lvef-prediction/XML dataset/"
    save_folder = "/home/work/.LVEF/ecg-lvef-prediction/ecg_visualizations_test/"

    # 폴더 내의 모든 XML 파일을 처리
    for filename in os.listdir(xml_folder):
        if filename.endswith('.xml'):
            filepath = os.path.join(xml_folder, filename)
            save_path = os.path.join(save_folder, f"{os.path.splitext(filename)[0]}_combined_leads.png")
            process_and_visualize_ecg(filepath, save_path)

# 이 스크립트가 직접 실행되었을 때만 main()을 실행
if __name__ == "__main__":
    main()
