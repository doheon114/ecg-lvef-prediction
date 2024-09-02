import matplotlib.pyplot as plt
import numpy as np

# Fold별 평가 메트릭
folds = {
    'fold1': {
        'int': {
            'accuracy': 0.619,
            'recall': 0.608,
            'precision': 0.7677,
            'specificity': 0.6406,
            'f1_score': 0.6786,
            'auroc': 0.6571,
            'auprc': 0.7787
        },
        'ext': {
            'accuracy': 0.6087,
            'recall': 0.4493,
            'precision': 0.8158,
            'specificity': 0.8478,
            'f1_score': 0.5794,
            'auroc': 0.7001,
            'auprc': 0.7934
        }
    },
    'fold2': {
        'int': {
            'accuracy': 0.6614,
            'recall': 0.664,
            'precision': 0.7905,
            'specificity': 0.6562,
            'f1_score': 0.7217,
            'auroc': 0.674,
            'auprc': 0.7964
        },
        'ext': {
            'accuracy': 0.6261,
            'recall': 0.4783,
            'precision': 0.825,
            'specificity': 0.8478,
            'f1_score': 0.6055,
            'auroc': 0.6957,
            'auprc': 0.8056
        }
    },
    'fold3': {
        'int': {
            'accuracy': 0.6349,
            'recall': 0.632,
            'precision': 0.7745,
            'specificity': 0.6406,
            'f1_score': 0.696,
            'auroc': 0.6866,
            'auprc': 0.7936
        },
        'ext': {
            'accuracy': 0.6609,
            'recall': 0.5072,
            'precision': 0.875,
            'specificity': 0.8913,
            'f1_score': 0.6422,
            'auroc': 0.7083,
            'auprc': 0.8229
        }
    },
    'fold4': {
        'int': {
            'accuracy': 0.6243,
            'recall': 0.6,
            'precision': 0.7812,
            'specificity': 0.6719,
            'f1_score': 0.6787,
            'auroc': 0.6748,
            'auprc': 0.7947
        },
        'ext': {
            'accuracy': 0.5913,
            'recall': 0.3768,
            'precision': 0.8667,
            'specificity': 0.913,
            'f1_score': 0.5253,
            'auroc': 0.6909,
            'auprc': 0.8116
        }
    },
    'fold5': {
        'int': {
            'accuracy': 0.6138,
            'recall': 0.624,
            'precision': 0.75,
            'specificity': 0.5938,
            'f1_score': 0.6812,
            'auroc': 0.6506,
            'auprc': 0.7931
        },
        'ext': {
            'accuracy': 0.6435,
            'recall': 0.4783,
            'precision': 0.8684,
            'specificity': 0.8913,
            'f1_score': 0.6168,
            'auroc': 0.6957,
            'auprc': 0.8035
        }
    }
}

# 평가 메트릭 이름
metrics = ['accuracy', 'recall', 'precision', 'specificity', 'f1_score', 'auroc', 'auprc']

# 각 fold에서의 메트릭을 수집
def get_fold_metrics(fold_data, metric):
    return [fold_data['int'][metric] for fold_data in folds.values()]

# 시각화
plt.figure(figsize=(15, 10))

for i, metric in enumerate(metrics):
    plt.subplot(3, 3, i+1)
    
    int_values = get_fold_metrics(folds, metric)
    ext_values = [folds[fold][dataset][metric] for fold in folds.keys() for dataset in ['ext']]
    
    bar_width = 0.35
    index = np.arange(len(folds))
    
    plt.bar(index - bar_width/2, int_values, bar_width, label='Internal', color='blue')
    plt.bar(index + bar_width/2, ext_values, bar_width, label='External', color='orange')
    
    plt.xlabel('Fold')
    plt.ylabel(metric.capitalize())
    plt.title(f'{metric.capitalize()} per Fold')
    plt.xticks(index, [f'Fold {i+1}' for i in index])
    plt.legend()

plt.tight_layout()
plt.savefig("evaluation_metrics_comparison.png")
plt.show()
