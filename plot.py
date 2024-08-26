import matplotlib.pyplot as plt

# Data for the four cases
depth = [ 4, 8, 16, 32, 64, 128, 256]
mean_mae_values = [0.107, 0.104, 0.113, 0.103, 0.126, 0.121, 0.135]
r2_scores = [0.094, 0.151, 0.003, 0.147, -0.251, -0.122, -0.400 ]

# Plotting Mean MAE values
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(depth, mean_mae_values, marker='o', linestyle='-', color='blue')
plt.title('Mean MAE vs batch_size')
plt.xlabel('batch_size')
plt.ylabel('Mean MAE')
plt.grid(True)

# Plotting R2 scores
plt.subplot(1, 2, 2)
plt.plot(depth, r2_scores, marker='o', linestyle='-', color='red')
plt.title('R2 Score vs batch_size')
plt.xlabel('batch_size')
plt.ylabel('R2 Score')
plt.grid(True)

# Show the plots
plt.tight_layout()
plt.savefig(f"results/compare/5.png")

plt.show()
