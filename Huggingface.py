from huggingface_hub import hf_hub_download
import os
root = os.getcwd()
import gc


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