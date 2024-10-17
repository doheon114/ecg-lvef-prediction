import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import os

# Define visualization path
visualization_path = "/home/work/.LVEF/ecg-lvef-prediction/ecg_visualizations_raw/"

# Function to parse ECG data from the XML format
def parse_ecg_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        ecg_content = f.read()

    # Splitting the waveform data based on the leads
    waveforms = ecg_content.split("<WaveFormData>")[1:]  # Gets all leads data
    leads = {
        "I": np.array(list(map(float, waveforms[0].split("<")[0].split(",")))),
        "II": np.array(list(map(float, waveforms[1].split("<")[0].split(",")))),
        "V1": np.array(list(map(float, waveforms[2].split("<")[0].split(",")))),
        "V2": np.array(list(map(float, waveforms[3].split("<")[0].split(",")))),
        "V3": np.array(list(map(float, waveforms[4].split("<")[0].split(",")))),
        "V4": np.array(list(map(float, waveforms[5].split("<")[0].split(",")))),
        "V5": np.array(list(map(float, waveforms[6].split("<")[0].split(",")))),
        "V6": np.array(list(map(float, waveforms[7].split("<")[0].split(",")))),
        "III": np.array(list(map(float, waveforms[8].split("<")[0].split(",")))),
        "aVR": np.array(list(map(float, waveforms[9].split("<")[0].split(",")))),
        "aVL": np.array(list(map(float, waveforms[10].split("<")[0].split(",")))),
        "aVF": np.array(list(map(float, waveforms[11].split("<")[0].split(","))))
    }
    return leads

# Function to clean and resample ECG signals
def clean_and_resample_ecg(ecg_leads):
    for lead, signal in ecg_leads.items():
        time_len = 10.0 if lead == "Rhythm strip" else 10
        # Keeping the signal as-is for now (commented out cleaning step)
        cleaned_signal = signal
        print(cleaned_signal.shape)
        # Resample the signal to 250 samples
        x = np.linspace(0, time_len, 5000, endpoint=False)
        xp = np.linspace(0, time_len, len(cleaned_signal), endpoint=False)
        ecg_leads[lead] = np.interp(x, xp, cleaned_signal)

    # Stack all 12 leads (I, II, III, aVR, aVL, aVF, V1-V6)
    ecg_stacked = np.stack([
        ecg_leads["I"], ecg_leads["II"], ecg_leads["III"],
        ecg_leads["aVR"], ecg_leads["aVL"], ecg_leads["aVF"],
        ecg_leads["V1"], ecg_leads["V2"], ecg_leads["V3"],
        ecg_leads["V4"], ecg_leads["V5"], ecg_leads["V6"]
    ]).T
    return ecg_stacked

# Function to visualize ECG data
def visualize_ecg(ecg_data, filename):
    plt.figure(figsize=(20, 15))
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    time = np.linspace(0, 10, ecg_data.shape[0]) 

    for i, lead in enumerate(leads):
        plt.subplot(12, 1, i + 1)
        plt.plot(time, ecg_data[:, i])
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(lead)
        plt.grid(True)

    os.makedirs(visualization_path, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_path, f"{filename}.png"))
    plt.close()

# Load, process, and visualize ECG from the provided XML file
filename = '/home/work/.LVEF/ecg-lvef-prediction/LBBB_dx/00000926/00000926_20220608145436_D.xml'
ecg_leads = parse_ecg_data(filename)
ecg_cleaned = clean_and_resample_ecg(ecg_leads)
visualize_ecg(ecg_cleaned, os.path.basename(filename).split('.')[0])
