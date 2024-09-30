import os
import pandas as pd
import torch

from fairseq_signals.utils.store import MemmapReader

root = os.getcwd()
print(root)
fairseq_signals_root = os.path.join(root, 'fairseq_signals/')
fairseq_signals_root = fairseq_signals_root.rstrip('/')
print(fairseq_signals_root)

# from huggingface_hub import hf_hub_download

# _ = hf_hub_download(
#     repo_id='wanglab/ecg-fm-preprint',
#     filename='physionet_finetuned.pt',
#     local_dir=os.path.join('/home/work/.LVEF/ecg-lvef-prediction/ECG-FM', 'ckpts'),
# )
# _ = hf_hub_download(
#     repo_id='wanglab/ecg-fm-preprint',
#     filename='physionet_finetuned.yaml',
#     local_dir=os.path.join('/home/work/.LVEF/ecg-lvef-prediction/ECG-FM', 'ckpts'),
# )

assert os.path.isfile(os.path.join('/home/work/.LVEF/ecg-lvef-prediction/ECG-FM', 'ckpts/physionet_finetuned.pt'))
assert os.path.isfile(os.path.join('/home/work/.LVEF/ecg-lvef-prediction/ECG-FM', 'ckpts/physionet_finetuned.yaml'))

segmented_split = pd.read_csv(
    os.path.join('/home/work/.LVEF/ecg-lvef-prediction/ECG-FM', 'data/code_15/segmented_split_incomplete.csv'),
    index_col='idx',
)
segmented_split['path'] = ('/home/work/.LVEF/ecg-lvef-prediction/ECG-FM' + '/data/code_15/segmented/') + segmented_split['path']
segmented_split.to_csv(os.path.join('/home/work/.LVEF/ecg-lvef-prediction/ECG-FM', 'data/code_15/segmented_split.csv'))
assert os.path.isfile(os.path.join('/home/work/.LVEF/ecg-lvef-prediction/ECG-FM', 'data/code_15/segmented_split.csv'))

# print(f"""cd {fairseq_signals_root}/scripts/preprocess
# python manifests.py \\
#     --split_file_paths "/home/work/.LVEF/ecg-lvef-prediction/ECG-FM/data/code_15/segmented_split.csv" \\
#     --save_dir "/home/work/.LVEF/ecg-lvef-prediction/ECG-FM/data/manifests/code_15_subset10/"
# """)

assert os.path.isfile(os.path.join('/home/work/.LVEF/ecg-lvef-prediction/ECG-FM', 'data/manifests/code_15_subset10/test.tsv'))

# print(f"""fairseq-hydra-inference \\
#     task.data="/home/work/.LVEF/ecg-lvef-prediction/ECG-FM/data/manifests/code_15_subset10/" \\
#     common_eval.path="/home/work/.LVEF/ecg-lvef-prediction/ECG-FM/ckpts/physionet_finetuned.pt" \\
#     common_eval.results_path="/home/work/.LVEF/ecg-lvef-prediction/ECG-FM/outputs" \\
#     model.num_labels=26 \\
#     dataset.valid_subset="test" \\
#     dataset.batch_size=10 \\
#     dataset.num_workers=3 \\
#     dataset.disable_validation=false \\
#     distributed_training.distributed_world_size=1 \\
#     distributed_training.find_unused_parameters=True \\
#     --config-dir "/home/work/.LVEF/ecg-lvef-prediction/ECG-FM/ckpts" \\
#     --config-name physionet_finetuned
# """)

assert os.path.isfile(os.path.join('/home/work/.LVEF/ecg-lvef-prediction/ECG-FM', 'outputs/outputs_test.npy'))
assert os.path.isfile(os.path.join('/home/work/.LVEF/ecg-lvef-prediction/ECG-FM', 'outputs/outputs_test_header.pkl'))

physionet2021_label_def = pd.read_csv(
    os.path.join('/home/work/.LVEF/ecg-lvef-prediction/ECG-FM', 'data/physionet2021/labels/label_def.csv'),
     index_col='name',
)
physionet2021_label_names = physionet2021_label_def.index
physionet2021_label_def

# Load the array of computed logits
logits = MemmapReader.from_header(
    os.path.join('/home/work/.LVEF/ecg-lvef-prediction/ECG-FM', 'outputs/outputs_test.npy')
)[:]
print(logits.shape)

# Construct predictions from logits
pred = pd.DataFrame(
    torch.sigmoid(torch.tensor(logits)).numpy(),
    columns=physionet2021_label_names,
)

# Join in sample information
pred = segmented_split.reset_index().join(pred, how='left').set_index('idx')
pred

# Perform a (crude) thresholding of 0.5 for all labels
pred_thresh = pred.copy()
pred_thresh[physionet2021_label_names] = pred_thresh[physionet2021_label_names] > 0.5

# Construct a readable column of predicted labels for each sample
pred_thresh['labels'] = pred_thresh[physionet2021_label_names].apply(
    lambda row: ', '.join(row.index[row]),
    axis=1,
)
pred_thresh['labels']

code_15_label_def = pd.read_csv(
    os.path.join('/home/work/.LVEF/ecg-lvef-prediction/ECG-FM', 'data/code_15/labels/label_def.csv'),
     index_col='name',
)
code_15_label_names = code_15_label_def.index
code_15_label_def

label_mapping = {
    'CRBBB|RBBB': 'RBBB',
    'CLBBB|LBBB': 'LBBB',
    'SB': 'SB',
    'STach': 'ST',
    'AF': 'AF',
}

physionet2021_label_def['name_mapped'] = physionet2021_label_def.index.map(label_mapping)
physionet2021_label_def

pred_mapped = pred.copy()
pred_mapped.drop(set(physionet2021_label_names) - set(label_mapping.keys()), axis=1, inplace=True)
pred_mapped.rename(label_mapping, axis=1, inplace=True)
pred_mapped

pred_thresh_mapped = pred_thresh.copy()
pred_thresh_mapped.drop(set(physionet2021_label_names) - set(label_mapping.keys()), axis=1, inplace=True)
pred_thresh_mapped.rename(label_mapping, axis=1, inplace=True)
pred_thresh_mapped['predicted'] = pred_thresh_mapped[label_mapping.values()].apply(
    lambda row: ', '.join(row.index[row]),
    axis=1,
)
pred_thresh_mapped

code_15_labels = pd.read_csv(os.path.join('/home/work/.LVEF/ecg-lvef-prediction/ECG-FM', 'data/code_15/labels/labels.csv'), index_col='idx')
code_15_labels['actual'] = code_15_labels[label_mapping.values()].apply(
    lambda row: ', '.join(row.index[row]),
    axis=1,
)
code_15_labels

# 예측된 레이블과 실제 레이블을 side-by-side로 시각화
result_df = pred_thresh_mapped[['predicted']].join(code_15_labels[['actual']], how='left')

# 결과를 CSV 파일로 저장
result_df.to_csv('/home/work/.LVEF/ecg-lvef-prediction/ECG-FM/results/predictions_vs_actuals.csv', index=False)

# 결과 확인
print("결과가 저장되었습니다:", '/home/work/.LVEF/ecg-lvef-prediction/ECG-FM/results/predictions_vs_actuals.csv')
