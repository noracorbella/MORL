import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull

# ── Data ─────────────────────────────────────────────────────────────────────
chvi_points = np.array([
    [ -1.8141,  -1.2005,  -1.7150],
    [ -2.3223,   0.0000,   0.0000],
    [ -2.0129,  -0.7143,   0.0000],
    [ -1.0100,  -4.0758,  -0.1801],
    [ -1.2285,  -3.9497,   0.0000],
    [ -1.7529,  -1.5204,   0.0000],
    [ -1.3815,  -3.2354,   0.0000],
    [ -1.2285,  -3.2354,  -0.1261],
    [ -1.0100,  -3.2354,  -0.6002],
    [ -1.6000,  -1.7887,  -0.1801],
    [ -0.8571,  -3.5037,  -0.7803],
    [ -2.2474,   0.0000,  -0.7350],
    [ -1.6000,  -1.2005,  -2.2331],
    [ -1.9059,   0.0000,  -1.2495],
    [ -0.8571,  -2.9155,  -2.3188],
    [ -1.1630,  -1.7150,  -1.3352],
    [ -0.4201,  -3.4300,  -1.9355],
    [ -0.3858,  -5.1709,  -0.8575],
    [ -0.2328,  -5.4392,  -1.0376],
    [  0.5163,  -6.7007,  -2.4500],
    [  0.8284,  -7.3010,  -2.4500],
    [ -1.7070,  -1.6207,  -1.8411],
    [ -0.7916,  -7.0000,   0.0000],
    [ -0.6386,  -7.0000,  -0.1261],
    [  0.5163, -16.7007,  -0.7350],
    [ -0.1673, -13.6505,   0.0000],
    [ -0.0144, -13.6505,  -0.1261],
    [  0.2041, -16.2058,  -0.1801],
    [ -0.0144, -16.0798,   0.0000],
    [  0.8284, -17.3010,  -0.7350],
])

lexhull_points = np.array([
    [-2.32,   0.00,   0.00],
    [-1.91,   0.00,  -1.25],
    [-0.01, -16.08,   0.00],
    [ 0.83, -17.30,  -0.73],
    [ 0.83,  -7.30,  -2.45],
])

CHVI_COLOR    = '#4fc3f7'
LEXHULL_COLOR = '#ff7043'
BG_COLOR      = '#0f0f1a'
PANEL_COLOR   = '#13131f'
GRID_COLOR    = '#1e1e35'

projections = [(0, 1), (0, 2), (1, 2)]
labels      = ['Obj 1', 'Obj 2', 'Obj 3']

def plot_hull_2d(ax, points, color, label, marker='o'):
    """Project points onto 2D, draw convex hull polygon and scatter."""
    hull = ConvexHull(points)
    # Close the polygon
    verts = np.append(hull.vertices, hull.vertices[0])
    ax.fill(points[hull.vertices, 0], points[hull.vertices, 1],
            alpha=0.15, color=color)
    ax.plot(points[verts, 0], points[verts, 1],
            color=color, linewidth=1.8, alpha=0.85)
    # Interior points (not on hull)
    interior = np.setdiff1d(np.arange(len(points)), hull.vertices)
    if len(interior):
        ax.scatter(points[interior, 0], points[interior, 1],
                   color=color, s=22, alpha=0.5, marker=marker, zorder=4)
    # Hull vertices
    ax.scatter(points[hull.vertices, 0], points[hull.vertices, 1],
               color=color, s=55, edgecolors='white', linewidth=0.7,
               marker=marker, zorder=5, label=label)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor(BG_COLOR)

for ax, (i, j) in zip(axes, projections):
    ax.set_facecolor(PANEL_COLOR)

    pts_chvi = chvi_points[:, [i, j]]
    pts_lex  = lexhull_points[:, [i, j]]

    plot_hull_2d(ax, pts_chvi, CHVI_COLOR,    f'CHVI ({len(chvi_points)} pts)')
    plot_hull_2d(ax, pts_lex,  LEXHULL_COLOR, f'LexHull ({len(lexhull_points)} pts)', marker='D')

    ax.set_xlabel(labels[i], color='#aaaacc', fontsize=11)
    ax.set_ylabel(labels[j], color='#aaaacc', fontsize=11)
    ax.set_title(f'{labels[i]} vs {labels[j]}', color='white',
                 fontsize=12, fontweight='bold', pad=10)
    ax.tick_params(colors='#aaaacc', labelsize=8)
    ax.grid(True, color=GRID_COLOR, linestyle='--', linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2a4a')
    ax.legend(fontsize=9, labelcolor='white',
              facecolor='#1a1a2e', edgecolor='#333355',
              loc='best', framealpha=0.85)

fig.suptitle('Convex Hull Comparison: CHVI vs LexHull — 2D Projections',
             color='white', fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(
    'hull_comparison_2d.png',
    dpi=150, bbox_inches='tight',
    facecolor=fig.get_facecolor(),
)
print("Saved → hull_comparison_2d.png")
plt.show()