import matplotlib.pyplot as plt
import numpy as np

# 케이스
cases = ['T/T', 'T/F', 'F/T', 'F/F']

# Mean MAE 값
mean_mae = [0.109, 0.119, 0.103, 0.111]

# R2 score 값
r2_scores = [0.086, -0.132, 0.147, 0.037]

# x축 위치 설정
x = np.arange(len(cases))  # 케이스 개수만큼 위치 생성

# 그래프 크기 설정
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# 막대 그래프 그리기 (Mean MAE)
bars1 = ax1.bar(x, mean_mae, color='skyblue', edgecolor='black')
ax1.set_xticks(x)
ax1.set_xticklabels(cases)
ax1.set_xlabel('Cases')
ax1.set_ylabel('Value')
ax1.set_title('Mean MAE by Case')

# 값 표시
def add_values(bars):
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

add_values(bars1)

# 막대 그래프 그리기 (R2 Score)
bars2 = ax2.bar(x, r2_scores, color='lightcoral', edgecolor='black')
ax2.set_xticks(x)
ax2.set_xticklabels(cases)
ax2.set_xlabel('Cases')
ax2.set_title('R2 Score by Case')

# 값 표시
def add_values(bars):
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

add_values(bars2)

# 레이아웃 조정
plt.tight_layout()
plt.savefig(f"results/compare/3.png")

plt.show()
