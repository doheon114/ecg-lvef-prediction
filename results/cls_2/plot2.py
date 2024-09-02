import matplotlib.pyplot as plt

# 데이터 설정
labels = ['T/T', 'T/F', 'F/T', 'F/F']
values = [0.6989, 0.6960, 0.6883, 0.6935]

# 막대 그래프 생성
plt.figure(figsize=(8, 6))
plt.bar(labels, values, color='skyblue')

# y축 범위 조정
plt.ylim(0.685, 0.7)

# 그래프 레이블 및 제목 설정
plt.xlabel('Conditions')
plt.ylabel('Values')
plt.title('Comparison of Values for Different Conditions')

# 각 막대 위에 값 표시
for i, v in enumerate(values):
    plt.text(i, v + 0.0002, f"{v:.4f}", ha='center', va='bottom')

# 그래프 보여주기
plt.savefig(f"results/8.png")

plt.show()
