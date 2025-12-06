import matplotlib.pyplot as plt
import numpy as np

# Data
alpha = np.array([-1, 0.9, 1.0, 1.1, 1.5])
top1 = np.array([72.41, 75.58, 75.78, 75.34, 75.54])
top5 = np.array([92.83, 93.95, 94.35, 94.07, 94.07])

# Create broken axis figure
fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1,
    sharex=True,
    figsize=(7, 5),
    gridspec_kw={'height_ratios': [0.8, 1.1]}  # match data ranges
)

# ======== Plot on both axes (so it looks like ONE dataset) ========
ax_top.plot(alpha, top5, marker='s', linestyle='--', color='lightgreen', linewidth=1.5, label='Top-5 Accuracy')
ax_bottom.plot(alpha, top1, marker='o', color='tab:green', linewidth=2.5, label='Top-1 Accuracy')

# ======== Set y-axis ranges (keep middle blank) ========
ax_top.set_ylim(92.8, 94.5)
ax_bottom.set_ylim(72.3, 76)

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
ax_top.legend(loc='best', fontsize=11)
ax_bottom.legend(loc='best', fontsize=11)
ax_top.grid(alpha=0.3)

plt.tight_layout(rect=[0.04, 0.0, 1, 0.95])
plt.savefig('./graphs/alpha_tuning_plot_broken.png', dpi=300)
plt.show()