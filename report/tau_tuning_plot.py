import matplotlib.pyplot as plt
import numpy as np

# Tau values (x-axis)
taus = np.array([0.05, 0.07, 0.35, 0.45, 0.5, 0.6, 1.0])

# Top-1 accuracy for each tau
top1 = np.array([75.55, 75.88, 75.83, 75.87, 76.01, 75.34, 75.41])

# Top-5 accuracy for each tau
top5 = np.array([94.19, 94.11, 94.12, 94.44, 94.54, 94.23, 94.19])

plt.figure(figsize=(7, 5))

# Plot Top-1 (main focus)
plt.plot(
    taus, top1, 
    marker='o', linewidth=2.5, markersize=7, 
    label='Top-1 Accuracy',
    color='tab:green'
)

# Plot Top-5 (lighter, secondary)
plt.plot(
    taus, top5, 
    marker='s', linewidth=1.5, markersize=6, 
    label='Top-5 Accuracy',
    color='lightgreen', linestyle='--'
)

# Labeling
plt.xlabel(r'Temperature $\tau$', fontsize=13)
plt.ylabel('Accuracy (%)', fontsize=13)
plt.title('Effect of Temperature $\\tau$ on SW-CRD Performance', fontsize=14)

plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()

plt.show()