
import matplotlib.pyplot as plt
import numpy as np

# 케이스
cases = ['T/T', 'T/F', 'F/T', 'F/F']

# Mean MAE 값
mean_mae = [0.121, 0.124, 0.122, 0.124]

# R2 score 값
r2_scores = [-0.139, -0.175, -0.122, -0.149 ]


# x축 위치 설정
x = np.arange(len(cases))  # 케이스 개수만큼 위치 생성

# 그래프 크기 설정
fig, ax = plt.subplots(figsize=(10, 6))

# 막대 그래프 그리기
bar_width = 0.35  # 막대의 폭

bars1 = ax.bar(x - bar_width/2, mean_mae, bar_width, label='Mean MAE')
bars2 = ax.bar(x + bar_width/2, r2_scores, bar_width, label='R2 Score')

# 레이블 및 제목 설정
ax.set_xlabel('Cases')
ax.set_ylabel('Values')
ax.set_title('Comparison of Mean MAE and R2 Score by Case')
ax.set_xticks(x)
ax.set_xticklabels(cases)
ax.legend()

# 값 표시
def add_values(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_values(bars1)
add_values(bars2)

# 레이아웃 조정
plt.tight_layout()
plt.savefig(f"results/compare/4.png")

plt.show()
