import os
import torch
import pandas as pd
from fairseq_signals.models import build_model_from_checkpoint

import sys
sys.setrecursionlimit(3000)  # 재귀 한도를 늘립니다.


# 1. 체크포인트 경로 설정
checkpoint_path = os.path.join('/home/work/.LVEF/ecg-lvef-prediction/ECG-FM', 'ckpts/physionet_finetuned.pt')

# 2. 모델 로드
model = build_model_from_checkpoint(checkpoint_path)
model.eval()  # 모델을 평가 모드로 설정

# 3. 데이터 준비
# CSV 파일에서 데이터 로드 (여기서 'path_to_your_data.csv'는 실제 데이터 파일의 경로로 변경)
data = pd.read_csv('path_to_your_data.csv')

# 예시로 첫 번째 열을 특성(X)로 사용하고 나머지를 타겟(y)로 사용한다고 가정
X = data.iloc[:, :-1].values  # 마지막 열을 제외한 모든 열
# y = data.iloc[:, -1].values  # 실제 라벨 (필요할 경우)

# 4. 예측 수행
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 데이터 텐서로 변환 및 이동
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

with torch.no_grad():  # 경량화된 예측을 위해 그래디언트 계산 비활성화
    predictions = model(X_tensor)

# 5. 예측 결과 해석
predictions_np = predictions.cpu().numpy()  # 결과를 NumPy 배열로 변환
# 필요에 따라 클래스 레이블이나 확률로 변환 (예: argmax 사용)
predicted_classes = predictions_np.argmax(axis=1)  # 다중 클래스 예측일 경우

# 6. 결과 저장
# 예측 결과를 CSV로 저장
results_df = pd.DataFrame({
    'Predictions': predicted_classes
    # 필요에 따라 추가적인 열을 포함할 수 있습니다.
})

results_df.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")
