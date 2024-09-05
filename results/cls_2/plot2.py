import matplotlib.pyplot as plt

# 데이터 설정
labels = ['T/T', 'T/F', 'F/T', 'F/F']
values = [0.6562, 0.6465, 0.6414, 0.6636]

# 막대 그래프 생성
plt.figure(figsize=(8, 6))
plt.bar(labels, values, color='skyblue')

# y축 범위 조정
plt.ylim(0.64, 0.68)

# 그래프 레이블 및 제목 설정
plt.xlabel('Conditions')
plt.ylabel('Values')
plt.title('Comparison of Values for Different Conditions')

# 각 막대 위에 값 표시
for i, v in enumerate(values):
    plt.text(i, v + 0.0002, f"{v:.4f}", ha='center', va='bottom')

# 그래프 보여주기
plt.savefig(f"results/residual&bottleneck.png")

plt.show()
