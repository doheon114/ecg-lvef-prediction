import matplotlib.pyplot as plt

# Data for the four cases
depth = [8, 16, 32, 64, 128, 256]
mean_accuracy_values = [0.6621, 0.6518, 0.6989, 0.6672, 0.7057, 0.705 ]

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
plt.savefig(f"results/10.png")

plt.show()
