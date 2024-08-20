import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm
import neurokit2 as nk
import pickle

# Define function

xml_folder = "/home/work/.LVEF/ecg-lvef-prediction/XML dataset"
raw_meta = pd.read_excel("/home/work/.LVEF/ecg-lvef-prediction/lbbb with LVEF(with duplicated_file, add phase).xlsx")

def XMLloader(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        ecg = f.read()
    tmp = ecg.split(">")
    ecg = {
        "I": np.array(list(map(float, tmp[4][:-10].split(" ")))),
        "II": np.array(list(map(float, tmp[8][:-11].split(" ")))),
        "III": np.array(list(map(float, tmp[12][:-12].split(" ")))),
        "aVR": np.array(list(map(float, tmp[16][:-12].split(" ")))),
        "aVL": np.array(list(map(float, tmp[20][:-12].split(" ")))),
        "aVF": np.array(list(map(float, tmp[24][:-12].split(" ")))),
        "V1": np.array(list(map(float, tmp[28][:-11].split(" ")))),
        "V2": np.array(list(map(float, tmp[32][:-11].split(" ")))),
        "V3": np.array(list(map(float, tmp[36][:-11].split(" ")))),
        "V4": np.array(list(map(float, tmp[40][:-11].split(" ")))),
        "V5": np.array(list(map(float, tmp[44][:-11].split(" ")))),
        "V6": np.array(list(map(float, tmp[48][:-11].split(" ")))),
        "Rhythm strip": np.array(list(map(float, tmp[52][:-17].split(" ")))),
    }

    return ecg 

def clean_ecg(ecg):
    
    for lead, signal in ecg.items() : 
        time_len = 10.0 if lead == "Rhythm strip" else 2.5
        fp = nk.ecg_clean(signal, sampling_rate=int(len(signal) / time_len))
        x = np.linspace(0, time_len, 1000 if lead == "Rhythm strip" else 250, endpoint=False)
        xp = np.linspace(0, time_len, len(fp), endpoint=False)
        ecg[lead] = np.interp(x, xp, fp)

    # shape: (4, 1000)
    ecg = np.stack([np.concatenate((ecg["I"], ecg["aVR"], ecg["V1"], ecg["V4"])),
                        np.concatenate((ecg["II"], ecg["aVL"], ecg["V2"], ecg["V5"])),
                        np.concatenate((ecg["III"], ecg["aVF"], ecg["V3"], ecg["V6"])),
                        ecg["Rhythm strip"]]).T
    
    return ecg

data = {}

for phase in ["train", "int test", "ext test"]:
    print(f"Phase {phase} processing...")
    ecg, label = [], []

    meta_info = raw_meta[raw_meta["phase"]== phase].reset_index(drop=True)
    id_ = meta_info['id'].astype(str) if phase == "ext test" else meta_info['id'].astype(str).str.zfill(8)

    id_to_lvef = dict(zip(id_, meta_info['LVEF']))

    file_paths = glob.glob(os.path.join(xml_folder, "*.xml"))
    for file in tqdm(file_paths):
        file_name = os.path.basename(file)
        file_id = file_name.split("_")[0]
        if file_id in id_to_lvef:
            ecg.append(clean_ecg(XMLloader(file)))
            label.append(id_to_lvef[file_id])

    data[phase] = {"x": np.stack(ecg, 0), "y": np.array(label)/100}

with open("/home/work/.LVEF/ecg-lvef-prediction/data/processed.pkl", "wb") as f:
    pickle.dump(data, f)