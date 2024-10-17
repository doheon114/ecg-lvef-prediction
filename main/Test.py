import random
import os 
import torch 
import numpy as np 

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

from moment.momentfm.data.ptbxl_classification_dataset import PTBXL_dataset
import torch 

class Config:
    # Path to the unzipped PTB-XL dataset folder
    basepath = '/Users/doheonkim/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3' # 'path/to/ptbxl_dataset'

    #path to cache directory to store preprocessed dataset if needed
    #note that preprocessing the dataset is time consuming so you might be benefited to cache it
    cache_dir = '/home/work/.LVEF/ecg-lvef-prediction/cache/' # 'path/to/cache_dir'
    load_cache = True

    #sampling frequency, choose from 100 or 500
    fs = 100

    # Class to predict
    code_of_interest = 'diagnostic_class'
    output_type = 'Single'

    #sequence length, only support 512 for now
    seq_len = 512

args = Config()

#create dataloader for training and testing
train_dataset = PTBXL_dataset(args, phase='train')
test_dataset = PTBXL_dataset(args, phase='test')
val_dataset = PTBXL_dataset(args, phase='val')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

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

test_embeddings, test_labels = get_timeseries(test_loader)
test_accuracy = clf.score(test_embeddings, test_labels)

print(f"Train accuracy: {train_accuracy:.2f}")
print(f"Test accuracy: {test_accuracy:.2f}")


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

test_embeddings, test_labels = get_embeddings(model, device, reduction, test_loader)
test_accuracy = clf.score(test_embeddings, test_labels)

print(f"Train accuracy: {train_accuracy:.2f}")
print(f"Test accuracy: {test_accuracy:.2f}")

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

epoch = 5
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=epoch * len(train_loader))
device = 'cuda:0'

for i in tqdm(range(epoch)):
    train_loss = train_epoch(model, device, train_loader, criterion, optimizer, scheduler)
    val_loss, val_accuracy = evaluate_epoch(val_loader, model, criterion, device, phase='test')
    print(f'Epoch {i}, train loss: {train_loss}, val loss: {val_loss}, val accuracy: {val_accuracy}')

test_loss, test_accuracy = evaluate_epoch(test_loader, model, criterion, device, phase='test')
print(f'Test loss: {test_loss}, test accuracy: {test_accuracy}')

#loading MOMENT with encoder unfrozen
from momentfm import MOMENTPipeline

model = MOMENTPipeline.from_pretrained(
                                        "AutonLab/MOMENT-1-large", 
                                        model_kwargs={
                                            'task_name': 'classification',
                                            'n_channels': 12,
                                            'num_class': 5,
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

epoch = 5
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, total_steps=epoch * len(train_loader))
device = 'cuda:0'

for i in tqdm(range(epoch)):
    train_loss = train_epoch(model, device, train_loader, criterion, optimizer, scheduler)
    val_loss, val_accuracy = evaluate_epoch(val_loader, model, criterion, device, phase='test')
    print(f'Epoch {i}, train loss: {train_loss}, val loss: {val_loss}, val accuracy: {val_accuracy}')

test_loss, test_accuracy = evaluate_epoch(test_loader, model, criterion, device, phase='test')
print(f'Test loss: {test_loss}, test accuracy: {test_accuracy}')

#set device to be 'cuda:0' or 'cuda' if you only have one GPU
device = 'cuda:0'
reduction = 'mean'
train_embeddings, train_labels = get_embeddings(model, device, reduction, train_loader)
clf = fit_svm(features=train_embeddings, y=train_labels)
train_accuracy = clf.score(train_embeddings, train_labels)

test_embeddings, test_labels = get_embeddings(model, device, reduction, test_loader)
test_accuracy = clf.score(test_embeddings, test_labels)

print(f"Train accuracy: {train_accuracy:.2f}")
print(f"Test accuracy: {test_accuracy:.2f}")