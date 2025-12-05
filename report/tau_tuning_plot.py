import matplotlib.pyplot as plt
import numpy as np

# Data
taus = np.array([0.05, 0.07, 0.35, 0.45, 0.5, 0.6, 1.0])
top1 = np.array([75.55, 75.88, 75.83, 75.87, 76.01, 75.34, 75.41])
top5 = np.array([94.19, 94.11, 94.12, 94.44, 94.54, 94.23, 94.19])

# Create broken axis figure
fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1,
    sharex=True,
    figsize=(7, 5),
    gridspec_kw={'height_ratios': [0.8, 1.1]}  # match data ranges
)

# ======== Plot on both axes (so it looks like ONE dataset) ========
ax_top.plot(taus, top5, marker='s', linestyle='--', color='lightgreen', linewidth=1.5)
ax_bottom.plot(taus, top1, marker='o', color='tab:green', linewidth=2.5)

# ======== Set y-axis ranges (keep middle blank) ========
ax_top.set_ylim(93.9, 94.7)
ax_bottom.set_ylim(75.2, 76.3)

# ======== Remove spines & add diagonal breaks ========
ax_top.spines['bottom'].set_visible(False)
ax_bottom.spines['top'].set_visible(False)
ax_top.tick_params(labelbottom=False)

d = 0.5
kwargs = dict(marker=[(-1,-d),(1,d)], markersize=12,
              linestyle='none', color='k', mec='k', mew=1, clip_on=False)

ax_top.plot([0,1],[0,0], transform=ax_top.transAxes, **kwargs)
ax_bottom.plot([0,1],[1,1], transform=ax_bottom.transAxes, **kwargs)

# ======== Labels ========
fig.text(0.02, 0.5, 'Accuracy (%)', va='center', rotation='vertical', fontsize=13)
ax_bottom.set_xlabel(r'Temperature $\tau$', fontsize=13)
# fig.suptitle(r'Effect of Temperature $\tau$ on SW-CRD Performance', fontsize=14)

ax_bottom.grid(alpha=0.3)
ax_top.grid(alpha=0.3)

plt.tight_layout(rect=[0.04, 0.0, 1, 0.95])
plt.savefig('./graphs/tau_tuning_plot_broken.png', dpi=300)
plt.show()