import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Data ────────────────────────────────────────────────────────────────────
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

# ── Hull computation ─────────────────────────────────────────────────────────
chvi_hull = ConvexHull(chvi_points)
print(len(chvi_points), len(chvi_hull.vertices))

lexhull_hull = ConvexHull(lexhull_points)
print(len(lexhull_points), len(lexhull_hull.vertices))

# ── Colors ───────────────────────────────────────────────────────────────────
CHVI_COLOR    = '#1a78c2'
LEXHULL_COLOR = '#d84315'
ALPHA_FACE    = 0.20
GRID_COLOR    = '#cccccc'

def add_hull(ax, points, hull, face_color, edge_color, label):
    verts = [points[simplex] for simplex in hull.simplices]
    poly  = Poly3DCollection(
        verts,
        alpha=ALPHA_FACE,
        facecolor=face_color,
        edgecolor=edge_color,
        linewidth=0.6,
    )
    ax.add_collection3d(poly)

    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        color=edge_color, s=28, zorder=5, depthshade=True,
        label=f'{label}',
    )

    hull_verts = points[hull.vertices]
    ax.scatter(
        hull_verts[:, 0], hull_verts[:, 1], hull_verts[:, 2],
        color=edge_color, s=70, edgecolors='black', linewidth=0.8,
        zorder=6, depthshade=False,
    )

def style_ax(ax, title):
    ax.set_facecolor('white')
    ax.set_title(title, color='black', fontsize=12, pad=10, fontweight='bold')
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = True
        pane.set_facecolor('#f5f5f5')
        pane.set_edgecolor('#cccccc')
    ax.tick_params(colors='black', labelsize=7)
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.zaxis.label.set_color('black')
    ax.set_xlabel('Obj 1', labelpad=4)
    ax.set_ylabel('Obj 2', labelpad=4)
    ax.set_zlabel('Obj 3', labelpad=4)
    ax.grid(True, color=GRID_COLOR, linestyle='--', linewidth=0.4)

# ── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 7))
fig.patch.set_facecolor('white')

# Left: CHVI only
ax1 = fig.add_subplot(131, projection='3d')
add_hull(ax1, chvi_points, chvi_hull, CHVI_COLOR, CHVI_COLOR, 'Convex Hull')
style_ax(ax1, f'Covex Hull - {len(chvi_points)} vectors')
# ax1.legend(loc='upper left', fontsize=8, facecolor='white', edgecolor='#cccccc')

# Middle: LexHull only
ax2 = fig.add_subplot(132, projection='3d')
add_hull(ax2, lexhull_points, lexhull_hull, LEXHULL_COLOR, LEXHULL_COLOR, 'LexHull')
style_ax(ax2, f'Lexicographic Hull - {len(lexhull_points)} vectors')
# ax2.legend(loc='upper left', fontsize=8, facecolor='white', edgecolor='#cccccc')

# Right: overlay
ax3 = fig.add_subplot(133, projection='3d')
add_hull(ax3, chvi_points,    chvi_hull,    CHVI_COLOR,    CHVI_COLOR,    'Convex Hull')
add_hull(ax3, lexhull_points, lexhull_hull, LEXHULL_COLOR, LEXHULL_COLOR, 'Lexicographic Hull')
style_ax(ax3, 'Overlay')
ax3.legend(loc='upper left', fontsize=8, facecolor='white', edgecolor='#cccccc')

# ── Title & info ─────────────────────────────────────────────────────────────
# fig.suptitle(
#     'Convex Hull Comparison: CHVI vs LexHull',
#     color='black', fontsize=15, fontweight='bold', y=1.01,
# )

chvi_vol = chvi_hull.volume
lex_vol  = lexhull_hull.volume
# info_text = (
#     f'CHVI    — {len(chvi_points)} vectors  |  vol ≈ {chvi_vol:.2f}\n'
#     f'LexHull — {len(lexhull_points)} vectors  |  vol ≈ {lex_vol:.2f}\n'
#     f'Volume ratio (CHVI / Lex) ≈ {chvi_vol / lex_vol:.1f}×'
# )
# fig.text(
#     0.5, -0.02, info_text,
#     ha='center', va='top', fontsize=9,
#     color='#333333', family='monospace',
#     bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', edgecolor='#cccccc'),
# )

plt.tight_layout()
plt.savefig(
    'hull_comparison_3d_white.png',
    dpi=150, bbox_inches='tight',
    facecolor='white',
)
print("Saved → hull_comparison_3d_white.png")
plt.show()