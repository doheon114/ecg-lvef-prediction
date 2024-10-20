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

X_train = torch.tensor(data["train"]["x"], dtype=torch.float32)
y_train = torch.tensor(data["train"]["y"], dtype=torch.float32)
X_int = torch.tensor(data["int test"]["x"], dtype=torch.float32)
y_int = torch.tensor(data["int test"]["y"], dtype=torch.float32)
X_ext = torch.tensor(data["ext test"]["x"], dtype=torch.float32)
y_ext = torch.tensor(data["ext test"]["y"], dtype=torch.float32)

# 이진 분류를 위한 타겟 설정 (0 또는 1)
th = 0.4
y_train_binary = (y_train >= th).float()
y_int_binary = (y_int >= th).float()
y_ext_binary = (y_ext >= th).float()



# 데이터 6:2로 train:val 나누기
train_size = int(0.6 * len(X_train))
val_size = len(X_train) - train_size
train_dataset, val_dataset = random_split(TensorDataset(X_train, y_train_binary), [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ClassificationHead 클래스 수정 (마지막에 Sigmoid 활성화 함수 추가)
class ClassificationHead(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)  # 최종 이진 분류 출력

    def forward(self, x):
        x = self.fc(x)
        return x

# Wav2Vec2CMSCModel에 classification head 추가
class FineTunedWav2Vec2Model(nn.Module):
    def __init__(self, pretrained_model, input_dim):
        super(FineTunedWav2Vec2Model, self).__init__()
        self.pretrained_model = pretrained_model
        self.classification_head = ClassificationHead(input_dim=input_dim)

    def forward(self, source):
        # pretrained 모델의 feature 추출
        outputs = self.pretrained_model(source=source)
        
        # features 값을 사용하여 classification head 통과
        final_proj_output = outputs['features']  # final_proj_output 크기: [batch_size, seq_len, feature_dim]
        
        # [batch_size, seq_len * feature_dim]으로 reshape
        final_proj_output = final_proj_output.reshape(final_proj_output.size(0), -1)  # (batch_size, seq_len * feature_dim)

        # classification head에 전달 (input_dim을 768 * 15로 설정)
        logits = self.classification_head(final_proj_output)
        return logits

# pretrained 모델에 classification head 붙이기
input_dim = 49152  # final_proj_output의 feature 크기가 768이고 시퀀스 길이가 15이므로 768 * 15로 설정
model_with_classification_head = FineTunedWav2Vec2Model(pretrained_model=model_pretrained, input_dim=input_dim)

# 다중 GPU 지원을 위해 nn.DataParallel 사용
if torch.cuda.device_count() > 1:
    model_with_classification_head = nn.DataParallel(model_with_classification_head)

# 모델, 손실 함수, 최적화기 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_with_classification_head.to(device)  # 모델을 GPU로 이동
optimizer = torch.optim.Adam(model_with_classification_head.parameters(), lr=1e-6, betas=(0.9, 0.98))
criterion = F.binary_cross_entropy_with_logits  # 이진 분류 손실 함수 사용

# Fine-tuning 과정에서 손실 값을 기록할 리스트
train_loss = []
val_accuracy_list = []

# Fine-tuning 과정
num_epochs = 50
for epoch in range(num_epochs):
    model_with_classification_head.train()
    running_loss = 0.0  # 에포크별 손실값 누적 변수
    for batch in tqdm(train_loader):
        inputs, labels = batch
        
        # 데이터도 GPU로 이동
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()  # 기울기 초기화
        outputs = model_with_classification_head(source=inputs)  # 모델 예측
        
        # 이진 분류를 위한 logits 추출
        logits = outputs.squeeze()  # (batch_size, 1)을 (batch_size,)로 변환
        
        # 손실 계산
        loss = criterion(logits, labels)
        loss.backward()  # 기울기 계산
        optimizer.step()  # 최적화
        
        running_loss += loss.item()  # 손실값 누적
    
    # Epoch별 평균 손실값 저장
    epoch_loss = running_loss / len(train_loader)
    train_loss.append(epoch_loss)
    
    # Validation 성능 평가
    model_with_classification_head.eval()
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
    
    # Validation Accuracy 계산
    val_accuracy = val_correct / val_total
    val_accuracy_list.append(val_accuracy)
    
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

# 학습 손실 그래프 그리기
plt.plot(np.arange(1, num_epochs + 1), train_loss, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig("/home/work/.LVEF/ecg-lvef-prediction/training_loss.png")  # 학습 손실 그래프 저장
plt.show()  # 그래프 보여주기

# Validation 정확도 그래프 그리기
plt.plot(np.arange(1, num_epochs + 1), val_accuracy_list, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.savefig("/home/work/.LVEF/ecg-lvef-prediction/validation_accuracy.png")  # 검증 정확도 그래프 저장
plt.show()  # 그래프 보여주기

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, ConfusionMatrixDisplay

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

    # 혼란 행렬
    cm_int = confusion_matrix(y_int_binary.cpu(), test_predictions.cpu())
    disp_int = ConfusionMatrixDisplay(cm_int, display_labels=[0, 1])
    disp_int.plot()
    plt.title("Confusion Matrix (Internal Test Set)")
    plt.savefig("/home/work/.LVEF/ecg-lvef-prediction/confusion_matrix_internal.png")  # 혼란 행렬 저장
    plt.close()  # 현재 그래프 닫기

    X_int = X_int.cpu()
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

    # 혼란 행렬
    cm_ext = confusion_matrix(y_ext_binary.cpu(), test_predictions.cpu())
    disp_ext = ConfusionMatrixDisplay(cm_ext, display_labels=[0, 1])
    disp_ext.plot()
    plt.title("Confusion Matrix (External Test Set)")
    plt.savefig("/home/work/.LVEF/ecg-lvef-prediction/confusion_matrix_external.png")  # 혼란 행렬 저장
    plt.close()  # 현재 그래프 닫기
