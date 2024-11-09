import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from fairseq_signals.models import build_model_from_checkpoint
from fairseq_signals.utils import checkpoint_utils

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import json
import gc


# GPU 및 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 모델 로드 및 초기 설정
root = os.getcwd()

# 체크포인트 다운로드
_ = hf_hub_download(
    repo_id='wanglab/ecg-fm-preprint',
    filename='mimic_iv_ecg_physionet_pretrained.pt',
    local_dir=os.path.join(root, 'ckpts'),
)
_ = hf_hub_download(
    repo_id='wanglab/ecg-fm-preprint',
    filename='mimic_iv_ecg_physionet_pretrained.yaml',
    local_dir=os.path.join(root, 'ckpts'),
)
_ = hf_hub_download(
    repo_id='wanglab/ecg-fm-preprint',
    filename='physionet_finetuned.pt',
    local_dir=os.path.join(root, 'ckpts'),
)
_ = hf_hub_download(
    repo_id='wanglab/ecg-fm-preprint',
    filename='physionet_finetuned.yaml',
    local_dir=os.path.join(root, 'ckpts'),
)

# # 모델 초기화
# model_pretrained = build_model_from_checkpoint(
#     checkpoint_path=os.path.join(root, 'ckpts/mimic_iv_ecg_physionet_pretrained.pt'), strict=False
# ).to(device)  # 모델을 GPU로 이동

# print(model_pretrained)
# 데이터 로드
with open("/home/work/.LVEF/ecg-lvef-prediction/data/processed_old_(12,5000).pkl", "rb") as f:
    data = pickle.load(f)
X_train = torch.tensor(data["train"]["x"], dtype=torch.float32)
y_train = torch.tensor(data["train"]["y"], dtype=torch.float32)
X_int = torch.tensor(data["int test"]["x"], dtype=torch.float32)
y_int = torch.tensor(data["int test"]["y"], dtype=torch.float32)
X_ext = torch.tensor(data["ext test"]["x"], dtype=torch.float32)
y_ext = torch.tensor(data["ext test"]["y"], dtype=torch.float32)

with open("/home/work/.LVEF/ecg-lvef-prediction/data/new_processed_(12,5000).pkl", "rb") as f:
    data = pickle.load(f)
X_train_new = torch.tensor(data["train"]["x"], dtype=torch.float32)
y_train_new = torch.tensor(data["train"]["y"], dtype=torch.float32)
X_int_real = torch.tensor(data["int test"]["x"], dtype=torch.float32)
y_int_real = torch.tensor(data["int test"]["y"], dtype=torch.float32)
X_ext_two = torch.tensor(data["ext test"]["x"], dtype=torch.float32)
y_ext_two = torch.tensor(data["ext test"]["y"], dtype=torch.float32)

# 데이터 결합
# 클래스 별 비율을 train, tune, int에 대해 맞춰주기 위해 concatenate 후 stratify로 진행
X_train = np.concatenate((X_train, X_int, X_train_new), axis=0)
y_train = np.concatenate((y_train, y_int, y_train_new), axis=0)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

print(X_train.shape)
# 최종 데이터 피클로 저장
final_data = {
    "X_train": X_train,
    "y_train": y_train,
    "X_int": X_int_real,
    "y_int": y_int_real,
    "X_ext": X_ext,
    "y_ext": y_ext,
    "X_ext_two": X_ext_two,
    "y_ext_two": y_ext_two
}

with open("/home/work/.LVEF/ecg-lvef-prediction/data/final_combined_data_padded.pkl", "wb") as f:
    pickle.dump(final_data, f)

# # 이진 타겟 설정 (0 또는 1)
# th = 0.4
# y_train_binary = (y_train >= th).float()
# y_int_binary = (y_int_real >= th).float()
# y_ext_binary = (y_ext >= th).float()
# y_ext_two_binary = (y_ext_two >= th).float()

# X_train, X_val, y_train_binary, y_val_binary = train_test_split(X_train, y_train_binary, test_size=0.1, random_state=42)

# # TensorDataset 생성
# train_dataset = TensorDataset(X_train, torch.tensor(y_train_binary, dtype=torch.float32))
# val_dataset = TensorDataset(X_val, torch.tensor(y_val_binary, dtype=torch.float32))
# train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=False)
# val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, drop_last=False)


# model_finetuned = build_model_from_checkpoint(
#     checkpoint_path=os.path.join('/home/work/.LVEF/ecg-lvef-prediction/results_new/emergency/checkpoint_best.pt')
# )
# print(model_finetuned)
# class ClassificationHead(nn.Module):
#     def __init__(self, input_dim, output_dim=1, dropout_rate=0.5):
#         super(ClassificationHead, self).__init__()
#         self.fc = nn.Linear(input_dim, output_dim)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.batch_norm = nn.BatchNorm1d(input_dim)  # Add BatchNorm1d
#     def forward(self, x):
#         x = self.batch_norm(x)  # Apply batch normalization
#         x = self.dropout(x)
#         x = self.fc(x)
#         return x
# class FineTunedWav2Vec2Model(nn.Module):
#     def __init__(self, pretrained_model, input_dim):
#         super(FineTunedWav2Vec2Model, self).__init__()
#         self.pretrained_model = pretrained_model
#         self.conv1d_first = nn.Conv1d(4, 4, kernel_size=1)
#         self.conv1d = nn.Conv1d(4, 12, kernel_size=1)
#         self.classification_head = ClassificationHead(input_dim=input_dim)
#     def forward(self, source):
#         source = self.conv1d_first(source)
#         source = self.conv1d(source)
#         # Pretrained model feature extraction
#         outputs = self.pretrained_model(source=source)
#         final_proj_output = outputs['features']  # Shape: [batch_size, seq_len, feature_dim]
#         # Reshape to [batch_size, seq_len * feature_dim]
#         final_proj_output = final_proj_output.reshape(final_proj_output.size(0), -1)  # (batch_size, seq_len * feature_dim)
#         # Pass to the classification head
#         logits = self.classification_head(final_proj_output)
#         return logits
# # pretrained 모델에 classification head 붙이기
# input_dim = 239616
# model_with_classification_head = FineTunedWav2Vec2Model(pretrained_model=model_pretrained, input_dim=input_dim)



# # summary(model_with_classification_head, input_size=(1, 4, 5000))


# # # 모델의 모든 파라미터를 동결
# # for param in model_with_classification_head.parameters():
# #     param.requires_grad = False

# # feature_extractor와 추가한 레이어만 학습 가능하도록 설정
# # for param in model_with_classification_head.pretrained_model.feature_extractor.parameters():
# #     param.requires_grad = False

# # for param in model_with_classification_head.conv1d_first.parameters():
# #     param.requires_grad = True

# # for param in model_with_classification_head.conv1d.parameters():
# #     param.requires_grad = True

# # for param in model_with_classification_head.batch_norm.parameters():
# #     param.requires_grad = True

# # for param in model_with_classification_head.classification_head.parameters():
# #     param.requires_grad = True


# def count_trainable_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# trainable_params = count_trainable_parameters(model_with_classification_head)
# print(f"Trainable parameters: {trainable_params}")

# if torch.cuda.device_count() > 1:
#     model_with_classification_head = nn.DataParallel(model_with_classification_head)
    

#     # #     # 손실 함수, 최적화기 설정
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model_with_classification_head.to(device)
#     optimizer = torch.optim.Adam(model_with_classification_head.parameters(), lr=5e-7, betas = (0.9, 0.98))
#     criterion = F.binary_cross_entropy_with_logits
#     # 학습
#     train_loss, val_loss, train_accuracy_list, val_accuracy_list = [], [], [], []
#     num_epochs = 100
#     for epoch in range(num_epochs):
#         model_with_classification_head.train()
#         running_loss = 0.0
#         correct_predictions = 0
#         total_samples = 0

#         for batch in tqdm(train_loader):
#             inputs, labels = batch
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             positive_count = labels.sum().item()
#             negative_count = len(labels) - positive_count
#             pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32).to(device) if positive_count > 0 else torch.tensor([1.0], dtype=torch.float32).to(device)
            
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model_with_classification_head(source=inputs)
#             logits = outputs.squeeze()
#             loss = criterion(logits, labels, pos_weight=pos_weight)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
                
#             # 정확도 계산
#             preds = (torch.sigmoid(logits) >= 0.5).float()
#             correct_predictions += (preds == labels).sum().item()
#             total_samples += labels.size(0)

#         epoch_loss = running_loss / len(train_loader)
#         train_accuracy = correct_predictions / total_samples
#         train_loss.append(epoch_loss)
#         train_accuracy_list.append(train_accuracy)


#         model_with_classification_head.eval()
#         val_running_loss, val_correct, val_total = 0.0, 0, 0
#         with torch.no_grad():
#             for val_batch in val_loader:
#                 val_inputs, val_labels = val_batch
#                 val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
#                 val_outputs = model_with_classification_head(source=val_inputs)
#                 val_logits = val_outputs.squeeze()
#                 val_predictions = (torch.sigmoid(val_logits) >= 0.5).float()
#                 val_correct += (val_predictions == val_labels).sum().item()
#                 val_total += val_labels.size(0)
#                 val_loss_value = criterion(val_logits, val_labels)
#                 val_running_loss += val_loss_value.item()

#         epoch_val_loss = val_running_loss / len(val_loader)
#         val_loss.append(epoch_val_loss)
#         val_accuracy = val_correct / val_total
#         val_accuracy_list.append(val_accuracy)

#         print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

# # 학습 및 검증 손실 시각화
# plt.figure(figsize=(12, 6))
# plt.plot(np.arange(1, num_epochs + 1), train_loss, label='Train Loss')
# plt.plot(np.arange(1, num_epochs + 1), val_loss, label='Validation Loss', linestyle='--')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.savefig("/home/work/.LVEF/ecg-lvef-prediction/training_validation_loss.png")
# plt.show()

# # 학습 및 검증 정확도 시각화
# plt.figure(figsize=(12, 6))
# plt.plot(np.arange(1, num_epochs + 1), train_accuracy_list, label='Training Accuracy', color='blue')
# plt.plot(np.arange(1, num_epochs + 1), val_accuracy_list, label='Validation Accuracy', color='orange')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()
# plt.savefig("/home/work/.LVEF/ecg-lvef-prediction/training_validation_accuracy_plot.png")
# plt.show()

# # model_with_classification_head.load_state_dict(torch.load('/home/work/.LVEF/ecg-lvef-prediction/results_new/Best_5e-7, 2layer(4,4&4,12)_0.2,th=0.4, without batch_norm&dropout/final_model_weights.pth'))

# # 검증 데이터로 평가
# model_with_classification_head.eval()

# def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
#     """
#     Confusion Matrix를 시각화하고 저장하는 함수
#     Args:
#         y_true (np.array): 실제 라벨
#         y_pred (np.array): 예측 라벨
#         class_names (list): 클래스 이름 리스트
#         title (str): 시각화 제목
#         save_path (str): 저장 경로
#     """
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     ConfusionMatrixDisplay.from_predictions(
#         y_true,
#         y_pred,
#         display_labels=class_names,
#         cmap=plt.cm.Blues,
#     )
#     plt.title(title)
#     plt.savefig(save_path)
#     plt.close()

# # 모델 가중치 저장 함수
# def save_final_model(model, save_path):
#     """
#     모델의 최종 가중치를 지정한 경로에 저장하는 함수
#     Args:
#         model (nn.Module): 학습된 모델
#         save_path (str): 가중치를 저장할 경로
#     """
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     torch.save(model.state_dict(), save_path)
#     print(f"마지막 에포크의 모델 가중치가 '{save_path}'에 저장되었습니다.")



# # 성능 평가 및 혼동 행렬 시각화를 포함한 함수
# def evaluate_model(model, X, y_binary, dataset_name, device):
#     X = X.to(device)
#     model.eval()
#     with torch.no_grad():
#         outputs = model(source=X)
#         logits = outputs.squeeze()
#         predictions = (torch.sigmoid(logits) >= 0.5).float()


#         tn, fp, fn, tp = confusion_matrix(y_binary.cpu(), predictions.cpu()).ravel()

       
#         accuracy = (tp + tn) / (tp + tn + fp + fn)
#         sensitivity = tn / (tn + fp) if (tn + fp) > 0 else 0
#         precision = tn / (tn+ fn) if (tn + fn) > 0 else 0
#         NPV = tp / (tp + fp) if (tp +fp) > 0 else 0
#         specificity = tp / (tp + fn) if (tp + fn) > 0 else 0
#         f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
#         auc_roc = roc_auc_score(y_binary.cpu(), torch.sigmoid(logits).cpu())
#         auprc = average_precision_score(y_binary.cpu(), torch.sigmoid(logits).cpu())

#         print(f"Accuracy ({dataset_name}): {accuracy:.4f}")
#         print(f"Sensitivity ({dataset_name}): {sensitivity:.4f}")
#         print(f"Precision ({dataset_name}): {precision:.4f}")
#         print(f"NPV ({dataset_name}): {NPV:.4f}")
#         print(f"Specificity({dataset_name}): {specificity:.4f}")
#         print(f"F1 Score ({dataset_name}): {f1:.4f}")
#         print(f"AUROC ({dataset_name}): {auc_roc:.4f}")
#         print(f"AUPRC ({dataset_name}): {auprc:.4f}")

#         # 혼동 행렬 시각화 및 저장
#         plot_confusion_matrix(
#             y_true=y_binary.cpu().numpy(),
#             y_pred=predictions.cpu().numpy(),
#             class_names=[0, 1],
#             title=f"Confusion Matrix ({dataset_name})",
#             save_path=f"/home/work/.LVEF/ecg-lvef-prediction/confusion_matrix_{dataset_name.lower()}.png"
#         )

#         # 성능 지표를 딕셔너리에 저장
#         return {
#             "Dataset": dataset_name,
#             "Accuracy": accuracy,
#             "Sensitivity": sensitivity,
#             "Precision": precision,
#             "NPV": NPV,
#             "Specificity": specificity,
#             "F1 Score": f1,
#             "AUROC": auc_roc,
#             "AUPRC": auprc
#         }

# save_final_model(model_with_classification_head, "/home/work/.LVEF/ecg-lvef-prediction/final_model_weights.pth")

# # 각 테스트 세트에 대한 평가 및 결과 저장
# def save_metrics_to_file(model, device):
#     results = []
#     # 각 테스트 세트에 대해 평가 수행 및 결과 저장
#     results.append(evaluate_model(model, X_int_real, torch.tensor(y_int_binary, dtype=torch.float32), "Internal Test Set", device))
#     results.append(evaluate_model(model, X_ext, torch.tensor(y_ext_binary, dtype=torch.float32), "External Test Set", device))
#     results.append(evaluate_model(model, X_ext_two, torch.tensor(y_ext_two_binary, dtype=torch.float32), "External Two Test Set", device))

#     # JSON 파일로 저장
#     with open("/home/work/.LVEF/ecg-lvef-prediction/test_set_metrics.json", "w") as f:
#         json.dump(results, f, indent=4)
#     print("성능 지표가 test_set_metrics.json 파일에 저장되었습니다.")

# # 함수 호출 예시
# save_metrics_to_file(model_with_classification_head, device)
