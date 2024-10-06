from fairseq_signals.models import build_model_from_checkpoint
import os 
from huggingface_hub import hf_hub_download

_ = hf_hub_download(
    repo_id='wanglab/ecg-fm-preprint',
    filename='physionet_finetuned.pt',
    local_dir=os.path.join('/home/work/.LVEF/ecg-lvef-prediction/ECG-FM/ckpts'),
)
_ = hf_hub_download(
    repo_id='wanglab/ecg-fm-preprint',
    filename='physionet_finetuned.yaml',
    local_dir=os.path.join('/home/work/.LVEF/ecg-lvef-prediction/ECG-FM/ckpts'),
)

model_finetuned = build_model_from_checkpoint(
    checkpoint_path=os.path.join('/home/work/.LVEF/ecg-lvef-prediction/ECG-FM/ckpts/physionet_finetuned.pt')
)
model_finetuned
# Run if the pretrained model hasn't already been downloaded
from huggingface_hub import hf_hub_download

_ = hf_hub_download(
    repo_id='wanglab/ecg-fm-preprint',
    filename='mimic_iv_ecg_physionet_pretrained.pt',
    local_dir=os.path.join('/home/work/.LVEF/ecg-lvef-prediction/ECG-FM/ckpts'),
)
_ = hf_hub_download(
    repo_id='wanglab/ecg-fm-preprint',
    filename='mimic_iv_ecg_physionet_pretrained.yaml',
    local_dir=os.path.join('/home/work/.LVEF/ecg-lvef-prediction/ECG-FM/ckpts'),
)

model_pretrained = build_model_from_checkpoint(
    checkpoint_path=os.path.join('/home/work/.LVEF/ecg-lvef-prediction/ECG-FM/ckpts/mimic_iv_ecg_physionet_pretrained.pt')
)
model_pretrained