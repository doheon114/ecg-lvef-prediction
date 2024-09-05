import matplotlib.pyplot as plt

# Data for the four cases
depth = [4, 8, 16, 32, 64, 128, 256]
mean_accuracy_values = [ 0.6681, 0.6468, 0.6842, 0.6636, 0.6545, 0.6618, 0.6643]

# Plotting Mean MAE values
plt.figure(figsize=(6, 6))

plt.subplot(1, 1, 1)
plt.plot(depth, mean_accuracy_values, marker='o', linestyle='-', color='blue')
plt.title('Mean Acc vs batch_size')
plt.xlabel('batch_size')
plt.ylabel('Mean Acc')
plt.grid(True)


# Show the plots
plt.tight_layout()
plt.savefig(f"results/batch_size.png")

plt.show()
