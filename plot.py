import matplotlib.pyplot as plt

# Data for the four cases
depth = [4, 8, 16, 32, 64, 128, 256]
mean_mae_values = [ 0.125, 0.128, 0.121, 0.121, 0.128, 0.128, 0.146 ]
r2_scores = [ -0.226, -0.255, -0.106, -0.139, -0.275, 0.230, -0.585]

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
