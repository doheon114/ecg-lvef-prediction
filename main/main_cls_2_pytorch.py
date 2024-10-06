from momentfm import MOMENTPipeline
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np 

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

th=0.4 

# 데이터 불러오기 및 차원 맞추기
with open("data/processed.pkl", "rb") as f:
    data = pickle.load(f)
    
    X_train = data["train"]["x"]  # (n_samples, 4, 1000)
    y_train = data["train"]["y"]
    X_int = data["int test"]["x"]
    y_int = data["int test"]["y"]
    X_ext = data["ext test"]["x"]
    y_ext = data["ext test"]["y"]

# 데이터 전처리 (스케일링)
scaler = StandardScaler()
# 각 데이터셋에 대해 (n_samples, 1000, 4) -> (n_samples, 4, 1000)로 변환 후 스케일링
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape).transpose(0, 2, 1)  # (n_samples, 4, 1000)
X_int = scaler.transform(X_int.reshape(-1, X_int.shape[-1])).reshape(X_int.shape).transpose(0, 2, 1)
X_ext = scaler.transform(X_ext.reshape(-1, X_ext.shape[-1])).reshape(X_ext.shape).transpose(0, 2, 1)

y_train_binary = (y_train >= th).astype(np.int64)
y_int_binary = (y_int >= th).astype(np.int64)
y_ext_binary = (y_ext >= th).astype(np.int64)


# Tensor 형태로 변환 (타입 맞추기)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_binary, dtype=torch.long)
X_int_tensor = torch.tensor(X_int, dtype=torch.float32)
y_int_tensor = torch.tensor(y_int_binary, dtype=torch.long)
X_ext_tensor = torch.tensor(X_ext, dtype=torch.float32)
y_ext_tensor = torch.tensor(y_ext_binary, dtype=torch.long)

print(y_train_tensor)

# TensorDataset 및 DataLoader 생성
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
int_test_dataset = TensorDataset(X_int_tensor, y_int_tensor)
ext_test_dataset = TensorDataset(X_ext_tensor, y_ext_tensor)


from torch.utils.data import random_split

# 전체 데이터셋의 길이 확인
dataset_size = len(train_dataset)

# 4:1 비율로 나누기 (80%:20%)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

# random_split을 사용하여 데이터셋을 나눔
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

# 각각의 DataLoader 생성
train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)

int_test_loader = DataLoader(int_test_dataset, batch_size=16, shuffle=False)
ext_test_loader = DataLoader(ext_test_dataset, batch_size=16, shuffle=False)

# 확인용 출력
print(f"Train Loader Sample Shape: {next(iter(train_loader))[0].shape}")
print(f"Ext Test Loader Sample Shape: {next(iter(val_loader))[0].shape}")
print(f"Int Test Loader Sample Shape: {next(iter(int_test_loader))[0].shape}")
print(f"Int Test Loader Sample Shape: {next(iter(ext_test_loader))[0].shape}")


# Confusion matrix를 시각화하는 함수
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['LVSD', 'No LVSD'])
    
    plt.figure(figsize=(5, 5))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(title)
    plt.savefig(f'/home/work/.LVEF/ecg-lvef-prediction/confusion_matrix/{title}.png')
    plt.show()

import random
import os 
import torch 
import numpy as np 


model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large", 
    model_kwargs={
        'task_name': 'classification',
        'n_channels':5, # number of input channels
        'num_class': 2,
        'freeze_encoder': True, # Freeze the patch embedding layer
        'freeze_embedder': True, # Freeze the transformer encoder
        'freeze_head': False, # The linear forecasting head must be trained
        ## NOTE: Disable gradient checkpointing to supress the warning when linear probing the model as MOMENT encoder is frozen
        'enable_gradient_checkpointing': False,
        # Choose how embedding is obtained from the model: One of ['mean', 'concat']
        # Multi-channel embeddings are obtained by either averaging or concatenating patch embeddings 
        # along the channel dimension. 'concat' results in embeddings of size (n_channels * d_model), 
        # while 'mean' results in embeddings of size (d_model)
        'reduction': 'mean',
    },
    # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
    )

model.init()
print(model)
# Number of parameters in the encoder
num_params = sum(p.numel() for p in model.encoder.parameters())
print(f"Number of parameters: {num_params}")


def control_randomness(seed: int = 42):
    """Function to control randomness in the code."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
control_randomness(42)


import torch
from torch.utils.data import DataLoader
from tqdm import tqdm 
import numpy as np 

def get_timeseries(dataloader: DataLoader, agg='mean'):
    '''
    We provide two aggregation methods to convert the 12-lead ECG (2-dimensional) to a 1-dimensional time-series for SVM training:
    - mean: average over all channels, result in [1 x seq_len] for each time-series
    - channel: concat all channels, result in [1 x seq_len * num_channels] for each time-series

    labels: [num_samples]
    ts: [num_samples x seq_len] or [num_samples x seq_len * num_channels]

    *note that concat all channels will result in a much larger feature dimensionality, thus making the fitting process much slower
    '''
    ts, labels = [], []

    with torch.no_grad():
        for batch_x, batch_labels in tqdm(dataloader, total=len(dataloader)):
            # [batch_size x 12 x 512]
            if agg == 'mean':
                batch_x = batch_x.mean(dim=1)
                ts.append(batch_x.detach().cpu().numpy())
            elif agg == 'channel':
                ts.append(batch_x.view(batch_x.size(0), -1).detach().cpu().numpy())
            labels.append(batch_labels)        

    ts, labels = np.concatenate(ts), np.concatenate(labels)
    return ts, labels

# Fit a SVM classifier on the concatenated raw ECG signals
from momentfm.models.statistical_classifiers import fit_svm

train_embeddings, train_labels = get_timeseries(train_loader, agg='mean')
clf = fit_svm(features=train_embeddings, y=train_labels)
train_accuracy = clf.score(train_embeddings, train_labels)

test_embeddings, test_labels = get_timeseries(int_test_loader)
test_accuracy = clf.score(test_embeddings, test_labels)

print(f"Train accuracy: {train_accuracy:.2f}")
print(f"Int Test accuracy: {test_accuracy:.2f}")


test_embeddings, test_labels = get_timeseries(ext_test_loader)
test_accuracy = clf.score(test_embeddings, test_labels)

print(f"Ext Test accuracy: {test_accuracy:.2f}")
# the learning rate should be smaller to guide the encoder to learn the task without forgetting the pre-trained knowledge
import torch
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm 
import numpy as np 
from momentfm.models.statistical_classifiers import fit_svm

def get_embeddings(model, device, reduction, dataloader: DataLoader):
    '''
    labels: [num_samples]
    embeddings: [num_samples x d_model]
    '''
    embeddings, labels = [], []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_x, batch_labels in tqdm(dataloader, total=len(dataloader)):
            # [batch_size x 12 x 512]
            batch_x = batch_x.to(device).float()
            # [batch_size x num_patches x d_model (=1024)]
            output = model(x_enc=batch_x, reduction=reduction) 
            #mean over patches dimension, [batch_size x d_model]
            embedding = output.embeddings.mean(dim=1)
            embeddings.append(embedding.detach().cpu().numpy())
            labels.append(batch_labels)        

    embeddings, labels = np.concatenate(embeddings), np.concatenate(labels)
    return embeddings, labels

#set device to be 'cuda:0' or 'cuda' if you only have one GPU
device = 'cuda:0'
reduction = 'mean'
train_embeddings, train_labels = get_embeddings(model, device, reduction, train_loader)
clf = fit_svm(features=train_embeddings, y=train_labels)
train_accuracy = clf.score(train_embeddings, train_labels)

test_embeddings, test_labels = get_embeddings(model, device, reduction, int_test_loader)
test_accuracy = clf.score(test_embeddings, test_labels)

print(f"Train accuracy: {train_accuracy:.2f}")
print(f"Int Test accuracy: {test_accuracy:.2f}")


test_embeddings, test_labels = get_embeddings(model, device, reduction, ext_test_loader)
test_accuracy = clf.score(test_embeddings, test_labels)

print(f"Ext Test accuracy: {test_accuracy:.2f}")

def train_epoch(model, device, train_dataloader, criterion, optimizer, scheduler, reduction='mean'):
    '''
    Train only classification head
    '''
    model.to(device)
    model.train()
    losses = []

    for batch_x, batch_labels in train_dataloader:
        optimizer.zero_grad()
        batch_x = batch_x.to(device).float()
        batch_labels = batch_labels.to(device)

        #note that since MOMENT encoder is based on T5, it might experiences numerical unstable issue with float16
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32):
            output = model(x_enc=batch_x, reduction=reduction)
            loss = criterion(output.logits, batch_labels)
        loss.backward()

        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    
    avg_loss = np.mean(losses)
    return avg_loss

def evaluate_epoch(dataloader, model, criterion, device, phase='val', reduction='mean'):
    model.eval()
    model.to(device)
    total_loss, total_correct = 0, 0

    with torch.no_grad():
        for batch_x, batch_labels in dataloader:
            batch_x = batch_x.to(device).float()
            batch_labels = batch_labels.to(device)

            output = model(x_enc=batch_x, reduction=reduction)
            loss = criterion(output.logits, batch_labels)
            total_loss += loss.item()
            total_correct += (output.logits.argmax(dim=1) == batch_labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / len(dataloader.dataset)
    return avg_loss, accuracy

from tqdm import tqdm
import numpy as np 

epoch = 100
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=epoch * len(train_loader))
device = 'cuda:0'

#loading MOMENT with encoder unfrozen
from momentfm import MOMENTPipeline

model = MOMENTPipeline.from_pretrained(
                                        "AutonLab/MOMENT-1-large", 
                                        model_kwargs={
                                            'task_name': 'classification',
                                            'n_channels': 4,
                                            'num_class': 2,
                                            'freeze_encoder': False,
                                            'freeze_embedder': False,
                                            'reduction': 'mean',
                                        },
                                        )
model.init()

# the learning rate should be smaller to guide the encoder to learn the task without forgetting the pre-trained knowledge
import torch
from tqdm import tqdm
import numpy as np

epoch = 20
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, total_steps=epoch * len(train_loader))
device = 'cuda:0'

for i in tqdm(range(epoch)):
    train_loss = train_epoch(model, device, train_loader, criterion, optimizer, scheduler)
    val_loss, val_accuracy = evaluate_epoch(val_loader, model, criterion, device, phase='test')
    print(f'Epoch {i}, train loss: {train_loss}, val loss: {val_loss}, val accuracy: {val_accuracy}')

test_loss, test_accuracy = evaluate_epoch(int_test_loader, model, criterion, device, phase='test')
print(f'Int Test loss: {test_loss}, Int test accuracy: {test_accuracy}')


test_loss, test_accuracy = evaluate_epoch(ext_test_loader, model, criterion, device, phase='test')
print(f'Ext Test loss: {test_loss}, Ext test accuracy: {test_accuracy}')


#set device to be 'cuda:0' or 'cuda' if you only have one GPU
device = 'cuda:0'
reduction = 'mean'
train_embeddings, train_labels = get_embeddings(model, device, reduction, train_loader)
clf = fit_svm(features=train_embeddings, y=train_labels)
train_accuracy = clf.score(train_embeddings, train_labels)

test_embeddings, test_labels = get_embeddings(model, device, reduction, int_test_loader)
test_accuracy = clf.score(test_embeddings, test_labels)


# Int test set에 대해 confusion matrix 시각화
y_pred_int_test = clf.predict(test_embeddings)
plot_confusion_matrix(test_labels, y_pred_int_test, title="Int Test Set Confusion Matrix")

print(f"Train accuracy: {train_accuracy:.2f}")
print(f"Int Test accuracy: {test_accuracy:.2f}")


test_embeddings, test_labels = get_embeddings(model, device, reduction, ext_test_loader)
test_accuracy = clf.score(test_embeddings, test_labels)


# Ext test set에 대해 confusion matrix 시각화
y_pred_ext_test = clf.predict(test_embeddings)
plot_confusion_matrix(test_labels, y_pred_ext_test, title="Ext Test Set Confusion Matrix")
print(f"Ext Test accuracy: {test_accuracy:.2f}")



