import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
n_points = 200
n_classes = 5
points_per_class = n_points // n_classes

# 五类不均匀分布
v1 = np.array([0.2, 1.3])
v2 = np.array([-1.3, 0.5])
v3 = np.array([-0.6, -1.1])
v4 = np.array([0.9, -0.8])
v5 = np.array([1.4, 0.4])
targets = [v1, v2, v3, v4, v5]

class_labels = np.repeat([0, 1, 2, 3, 4], points_per_class)

x_start = np.random.normal(loc=0.0, scale=0.9, size=(n_points, 2))

x_target = np.zeros_like(x_start)
for i in range(n_points):
    cls = class_labels[i]
    x_target[i] = targets[cls] + np.random.normal(loc=0.0, scale=0.08, size=2)

def get_positions_at_t(t):
    return (1.0 - t) * x_start + t * x_target

time_steps = [0.00, 0.50, 1.00]
positions_by_t = {t: get_positions_at_t(t) for t in time_steps}

# 垂直三面板
fig, axes = plt.subplots(3, 1, figsize=(5, 12), dpi=150)
plt.subplots_adjust(hspace=0.35)

xlim, ylim = (-2.5, 2.5), (-2.5, 2.5)

grid_x = np.linspace(xlim[0], xlim[1], 120)
grid_y = np.linspace(ylim[0], ylim[1], 120)
X, Y = np.meshgrid(grid_x, grid_y)
grid_positions = np.vstack([X.ravel(), Y.ravel()])

for idx, t in enumerate(time_steps):
    ax = axes[idx]
    pos = positions_by_t[t]

    # KDE：大带宽 → 弥散
    kde = stats.gaussian_kde(pos.T, bw_method=0.25)
    Z = np.reshape(kde(grid_positions).T, X.shape)
    # 更多等高线层级，视觉占比更大
    ax.contour(X, Y, Z, levels=15, colors='#5DADE2', alpha=0.45, linewidths=0.8)

    ax.scatter(pos[:, 0], pos[:, 1], color='#5DADE2', s=12, alpha=0.85, edgecolors='none', zorder=3)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"$t = {t:.2f}$", fontsize=13, color='#555555')
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
        spine.set_linewidth(1.0)

plt.savefig("ELF_conceptual_illustration.svg", bbox_inches='tight', transparent=True)
plt.savefig("ELF_conceptual_illustration.pdf", bbox_inches='tight', transparent=True)
plt.show()
