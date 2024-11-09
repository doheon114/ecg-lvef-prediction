import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from fairseq_signals.models import build_model_from_checkpoint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, ConfusionMatrixDisplay

# 모델 로드 및 초기 설정
root = os.getcwd()

model_pretrained = build_model_from_checkpoint(
    checkpoint_path=os.path.join(root, 'ckpts/mimic_iv_ecg_physionet_pretrained.pt')
)

# processed.pkl 파일에서 데이터 불러오기
with open("/home/work/.LVEF/ecg-lvef-prediction/data/processed_echo.pkl", "rb") as f:
    data = pickle.load(f)
#X_train_int는 (train, tuning, int에 할당되는 데이터를 모두 모아놓았음. 클래스 별 비율을 동일하게 해서 나눠주기 위함.)
X_train_int = torch.tensor(data["train"]["x"], dtype=torch.float32)
y_train_int = torch.tensor(data["train"]["y"], dtype=torch.float32)

X_ext = torch.tensor(data["ext test"]["x"], dtype=torch.float32)
y_ext = torch.tensor(data["ext test"]["y"], dtype=torch.float32)
print(y_ext.shape)
print(X_ext.shape)

# 이진 분류를 위한 타겟 설정 (0 또는 1)
th = 0.4
y_train_int_binary = (y_train_int >= th).float()
y_ext_binary = (y_ext >= th).float()

from sklearn.model_selection import train_test_split

# 데이터 나누기 (6:2:2 비율, stratify로 클래스 비율 유지)
X_temp, X_int, y_temp_binary, y_int_binary = train_test_split(X_train_int, y_train_int_binary, test_size=0.2, stratify=y_train_int_binary, random_state=42)  # 20%를 internal test set으로
X_train, X_val, y_train_binary, y_val_binary = train_test_split(X_temp, y_temp_binary, test_size=0.25, stratify=y_temp_binary, random_state=42)  # 나머지 80% 중 25%를 validation set으로

print(f'Train shape: {X_train.shape}, Val shape: {X_val.shape}, Internal test shape: {X_int.shape}')

# TensorDataset을 사용해 X_train과 y_train_binary를 결합
train_dataset = TensorDataset(X_train, y_train_binary)
val_dataset = TensorDataset(X_val, y_val_binary)
# DataLoader로 변환
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, output_dim=1, dropout_rate=0.5):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm = nn.BatchNorm1d(input_dim)  # 배치 정규화를 추가

    def forward(self, x):
        x = self.batch_norm(x)  # 배치 정규화 적용
        x = self.dropout(x)
        x = self.fc(x)
        return x


class FineTunedWav2Vec2Model(nn.Module):
    def __init__(self, pretrained_model, input_dim):
        super(FineTunedWav2Vec2Model, self).__init__()
        self.pretrained_model = pretrained_model
        self.batch_norm = nn.BatchNorm1d(input_dim)  # 특징 추출 후 배치 정규화 추가
        self.classification_head = ClassificationHead(input_dim=input_dim)

    def forward(self, source):
        # 사전 학습된 모델의 특징 추출
        outputs = self.pretrained_model(source=source)
        final_proj_output = outputs['features']  # Shape: [batch_size, seq_len, feature_dim]

        # [batch_size, seq_len * feature_dim] 형태로 변환
        final_proj_output = final_proj_output.reshape(final_proj_output.size(0), -1)  # (batch_size, seq_len * feature_dim)

        # 배치 정규화 적용 후 classification head로 전달
        final_proj_output = self.batch_norm(final_proj_output)
        logits = self.classification_head(final_proj_output)
        return logits

# pretrained 모델에 classification head 붙이기
input_dim = 119808  
model_with_classification_head = FineTunedWav2Vec2Model(pretrained_model=model_pretrained, input_dim=input_dim)

# 다중 GPU 지원을 위해 nn.DataParallel 사용
if torch.cuda.device_count() > 1:
    model_with_classification_head = nn.DataParallel(model_with_classification_head)

# 모델, 손실 함수, 최적화기 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_with_classification_head.to(device)  # 모델을 GPU로 이동

# L2 정규화(weight_decay)는 Adam 옵티마이저 내에서 사용
optimizer = torch.optim.Adam(model_with_classification_head.parameters(), lr=2e-6, betas=(0.9, 0.98), weight_decay=1e-4)

criterion = F.binary_cross_entropy_with_logits  # 이진 분류 손실 함수 사용

# Fine-tuning 과정에서 손실 값을 기록할 리스트
train_loss = []
val_loss = []  # Validation loss 리스트 추가
val_accuracy_list = []

# Fine-tuning 과정
num_epochs = 40
for epoch in range(num_epochs):
    model_with_classification_head.train()
    running_loss = 0.0  # 에포크별 손실값 누적 변수
    for batch in tqdm(train_loader):
        inputs, labels = batch

        # 배치에서 클래스 비율 계산
        positive_count = labels.sum().item()
        negative_count = len(labels) - positive_count
        
        # 양성 클래스가 존재하는 경우에만 pos_weight 계산
        if positive_count > 0:
            pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32).to(device)
        else:
            pos_weight = torch.tensor([1.0], dtype=torch.float32).to(device)  # 양성 클래스가 없는 경우 기본값 1로 설정

        # 데이터도 GPU로 이동
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # 기울기 초기화
        outputs = model_with_classification_head(source=inputs)  # 모델 예측
        
        # 이진 분류를 위한 logits 추출
        logits = outputs.squeeze()  # (batch_size, 1)을 (batch_size,)로 변환
        
        # 배치별로 동적 pos_weight를 적용한 손실 함수 계산
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

        loss.backward()  # 기울기 계산
        optimizer.step()  # 최적화
        
        running_loss += loss.item()  # 손실값 누적

    # Epoch별 평균 손실값 저장
    epoch_loss = running_loss / len(train_loader)
    train_loss.append(epoch_loss)

    # Validation 성능 평가
    model_with_classification_head.eval()
    val_running_loss = 0.0  # Validation 손실값 누적 변수
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for val_batch in val_loader:
            val_inputs, val_labels = val_batch
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            
            val_outputs = model_with_classification_head(source=val_inputs)
            val_logits = val_outputs.squeeze()
            val_predictions = (torch.sigmoid(val_logits) >= 0.5).float()  # Sigmoid를 사용해 확률로 변환
            
            val_correct += (val_predictions == val_labels).sum().item()
            val_total += val_labels.size(0)

            # Validation 손실 계산
            val_loss_value = criterion(val_logits, val_labels)
            val_running_loss += val_loss_value.item()  # Validation 손실값 누적
    
    # Epoch별 평균 Validation 손실값 저장
    epoch_val_loss = val_running_loss / len(val_loader)
    val_loss.append(epoch_val_loss)

    # Validation Accuracy 계산
    val_accuracy = val_correct / val_total
    val_accuracy_list.append(val_accuracy)
    
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

# 학습 손실 및 Validation 손실 그래프 그리기
plt.figure(figsize=(12, 6))
plt.plot(np.arange(1, num_epochs + 1), train_loss, label='Train Loss')
plt.plot(np.arange(1, num_epochs + 1), val_loss, label='Validation Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig("/home/work/.LVEF/ecg-lvef-prediction/training_validation_loss.png")  # 학습 및 검증 손실 그래프 저장
plt.show()  # 그래프 보여주기

# Validation 정확도 그래프 그리기
plt.plot(np.arange(1, num_epochs + 1), val_accuracy_list, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.savefig("/home/work/.LVEF/ecg-lvef-prediction/validation_accuracy.png")  # 검증 정확도 그래프 저장
plt.show()  # 그래프 보여주기

# 검증 데이터로 평가
model_with_classification_head.eval()
with torch.no_grad():
    # Internal Test Set
    X_int = X_int.to(device)  # 데이터를 GPU로 이동
    test_outputs = model_with_classification_head(source=X_int)
    
    # 이진 분류 logits
    test_logits = test_outputs.squeeze()
    
    # 예측 확률을 0.5로 이진화
    test_predictions = (test_logits >= 0.5).float() 

    # 정확도 계산
    accuracy = (test_predictions == y_int_binary.to(device)).float().mean()
    print(f'Test Accuracy (Internal Test Set): {accuracy:.4f}')

    # AUROC 및 AUPRC 계산
    int_auc_roc = roc_auc_score(y_int_binary.cpu(), torch.sigmoid(test_logits).cpu())
    int_auprc = average_precision_score(y_int_binary.cpu(), torch.sigmoid(test_logits).cpu())
    print(f'AUROC (Internal Test Set): {int_auc_roc:.4f}')
    print(f'AUPRC (Internal Test Set): {int_auprc:.4f}')

    cm_int = confusion_matrix(y_int_binary.cpu(), test_predictions.cpu())
    disp_int = ConfusionMatrixDisplay(cm_int, display_labels=[0, 1])
    disp_int.plot()
    plt.title("Confusion Matrix (Internal Test Set)")
    plt.savefig("/home/work/.LVEF/ecg-lvef-prediction/confusion_matrix_internal.png")  
    plt.close()  # 현재 그래프 닫기

    # External Test Set
    X_ext = X_ext.to(device)  # 데이터를 GPU로 이동
    test_outputs = model_with_classification_head(source=X_ext)
    
    # 이진 분류 logits
    test_logits = test_outputs.squeeze()
    
    # 예측 확률을 0.5로 이진화
    test_predictions = (test_logits >= 0.5).float()
    
    # 정확도 계산
    accuracy = (test_predictions == y_ext_binary.to(device)).float().mean()
    print(f'Test Accuracy (External Test Set): {accuracy:.4f}')

    # AUROC 및 AUPRC 계산
    ext_auc_roc = roc_auc_score(y_ext_binary.cpu(), torch.sigmoid(test_logits).cpu())
    ext_auprc = average_precision_score(y_ext_binary.cpu(), torch.sigmoid(test_logits).cpu())
    print(f'AUROC (External Test Set): {ext_auc_roc:.4f}')
    print(f'AUPRC (External Test Set): {ext_auprc:.4f}')

    cm_ext = confusion_matrix(y_ext_binary.cpu(), test_predictions.cpu())
    disp_ext = ConfusionMatrixDisplay(cm_ext, display_labels=[0, 1])
    disp_ext.plot()
    plt.title("Confusion Matrix (External Test Set)")
    plt.savefig("/home/work/.LVEF/ecg-lvef-prediction/confusion_matrix_external.png")  # 혼란 행렬 저장
    plt.close()  # 현재 그래프 닫기