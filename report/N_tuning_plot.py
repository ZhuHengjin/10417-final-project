import matplotlib.pyplot as plt
import numpy as np

# ===================== Data =====================
K = np.array([64, 256, 1024, 4096, 16384])
top1 = np.array([74.63, 75.08, 75.63, 75.74, 76.01])
top5 = np.array([93.75, 94.03, 94.27, 93.96, 94.54])

# ===================== Figure with Broken Axis =====================
fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1,
    sharex=True,
    figsize=(7, 5),
    gridspec_kw={'height_ratios': [1.1, 1.8]}
)

# Top-5 (dashed, light green) on upper axis
ax_top.plot(K, top5, marker='s', linestyle='--',
            color='lightgreen', linewidth=1.5, label='Top-5 Accuracy')

# Top-1 (solid, darker green) on lower axis
ax_bottom.plot(K, top1, marker='o',
               color='tab:green', linewidth=2.5, label='Top-1 Accuracy')

# ===================== Y-ranges =====================
ax_top.set_ylim(93.6, 94.7)
ax_bottom.set_ylim(74.4, 76.2)

# ===================== Broken axis effects =====================
ax_top.spines['bottom'].set_visible(False)
ax_bottom.spines['top'].set_visible(False)
ax_top.tick_params(labelbottom=False)

d = 0.5
kwargs = dict(marker=[(-1,-d), (1,d)], markersize=12,
              linestyle='none', color='k', mec='k', mew=1, clip_on=False)
ax_top.plot([0,1], [0,0], transform=ax_top.transAxes, **kwargs)
ax_bottom.plot([0,1], [1,1], transform=ax_bottom.transAxes, **kwargs)

# ===================== Labels =====================
fig.text(0.02, 0.5, 'Accuracy (%)', va='center', rotation='vertical', fontsize=13)
ax_bottom.set_xlabel('Number of Negatives K', fontsize=13)

ax_top.grid(alpha=0.3)
ax_bottom.grid(alpha=0.3)

ax_top.legend(loc='best', fontsize=11)
ax_bottom.legend(loc='best', fontsize=11)

plt.tight_layout(rect=[0.04, 0.0, 1, 0.95])
plt.savefig('./graphs/N_tuning_plot_broken.png', dpi=300)
plt.show()