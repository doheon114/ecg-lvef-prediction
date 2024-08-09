import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from scipy import signal
from scipy.signal import resample
from tqdm import tqdm
import neurokit2 as nk

# Define function
class DataLoader:
    def __init__(self, xml_folder, csv_file, phase, types, n_splits=5, random_state=None):
        self.xml_folder = xml_folder
        self.csv_file = csv_file
        self.n_splits = n_splits
        self.random_state = random_state
        self.types = types
        self.phase = phase

        self.ecg_data = []
        self.labels = []
        self._load_data()
        
    def _load_data(self):
        # CSV 파일을 읽고 ID와 LVEF 값을 매칭
        data = pd.read_excel(self.csv_file)
        data = data[data["phase"]== self.phase].reset_index(drop=True)
        if self.phase == "ext test" : 
            id_ = data['id'].astype(str)
        else :
            id_ = data['id'].astype(str).str.zfill(8)
        id_to_lvef = dict(zip(id_, data['LVEF']))
        
        # XML 파일 경로 패턴에 맞는 파일 경로를 찾기
        file_paths = glob.glob(os.path.join(self.xml_folder, "*.xml"))
        for file in file_paths:
            file_name = os.path.basename(file)
            file_id = file_name.split("_")[0]
            
            if file_id in id_to_lvef:
                ecg_dict = self.XMLloader(file)
                self.ecg_data.append(ecg_dict)
                
                lvef_value = id_to_lvef[file_id]
                self.labels.append(lvef_value)

        #for file in file_paths:
        #    file_name = os.path.basename(file)
        #    file_id = file_name.split("_")[0]
            
        #    if file_id in id_to_lvef:
        #        ecg_dict = self.XMLloader(file)
        #        self.ecg_data.append(ecg_dict)
                
        #       lvef_value = id_to_lvef[file_id]
        #        if self.types == "cls_2" :
        #            label_value = 0 if lvef_value < 40 else 1
        #            self.labels.append(label_value)
        #        elif self.types == "cls_3" :
        #            label_value = 0 if lvef_value < 40 else (1 if lvef_value < 60 else 2)
        #            self.labels.append(label_value)
        #        else :
        #            self.labels.append(lvef_value)
            #else:
                #print(f"LVEF value not found for ID: {file_id}")

    @staticmethod
    def XMLloader(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            ecg = f.read()  # 파일 내용을 읽어옴
        tmp = ecg.split(">")
        ecg_dict = {
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
        # for name in ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6", "Rhythm strip"] :
        #     print(ecg_dict[name].shape)
        return ecg_dict 
    
    
    def count_ecg_lengths(filename):
        length_counts = {}
        ecg_dict = XMLloader(filename)
        for key, array in ecg_dict.items():
            length = len(array)
            if length in length_counts:
                length_counts[length] += 1
            else:
                length_counts[length] = 1
        return length_counts

    
    def resample_and_pad(self, ecg_data, target_length=4096, target_rate=400):
        
        for lead, signal in ecg_data.items() : 
            time_len = 10.0 if lead == "Rhythm strip" else 2.5
            #fp = nk.ecg_clean(signal, sampling_rate=int(len(signal) / time_len))
            fp = signal
            x = np.linspace(0, time_len, 1000 if lead == "Rhythm strip" else 250, endpoint=False)
            xp = np.linspace(0, time_len, len(fp), endpoint=False)
            ecg_data[lead] = np.interp(x, xp, fp)

        # ecg_data = np.stack([np.concatenate((ecg_data["I"], ecg_data["aVR"], ecg_data["V1"], ecg_data["V4"])),
        #                     np.concatenate((ecg_data["II"], ecg_data["aVL"], ecg_data["V2"], ecg_data["V5"])),
        #                     np.concatenate((ecg_data["III"], ecg_data["aVF"], ecg_data["V3"], ecg_data["V6"])),
        #                     ecg_data["Rhythm strip"]])
        ecg_data = np.concatenate((ecg_data["I"], ecg_data["aVR"], ecg_data["V1"], ecg_data["V4"], ecg_data["II"], ecg_data["aVL"], ecg_data["V2"], ecg_data["V5"],
                                    ecg_data["III"], ecg_data["aVF"], ecg_data["V3"], ecg_data["V6"])).reshape(-1,250,12)
        
        return ecg_data
    
    def get_fold_data(self, fold_index):
        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=False)
        
        
        for i, (train_index, val_index) in tqdm(enumerate(kf.split(self.ecg_data, self.labels))):
            if i == fold_index:
                X_train = [self.ecg_data[idx] for idx in train_index]
                y_train = [self.labels[idx] for idx in train_index]
                X_val = [self.ecg_data[idx] for idx in val_index]
                y_val = [self.labels[idx] for idx in val_index]
                
                X_train = [self.resample_and_pad(ecg) for ecg in X_train]
                X_val = [self.resample_and_pad(ecg) for ecg in X_val]
                
                return (np.array(X_train).reshape(-1, 250,12), y_train), (np.array(X_val).reshape(-1, 250,12), y_val)
        
        raise IndexError(f"Fold index {fold_index} out of range for {self.n_splits} folds.")
    
    def get_test_data(self) :
        X_test = self.ecg_data
        y_test = self.labels
        return (X_test, y_test)



def show_batch(ecgs, label) :
    return "good"
