import matplotlib.pyplot as plt

# Data for the four cases
depth = [4, 8, 16, 32, 64, 128]
mean_mae_values = [0.139, 0.133, 0.119, 0.112, 0.125, 0.127 ]
r2_scores = [ -0.536, -0.383, -0.078, -0.135, -0.221, -0.287 ]

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
