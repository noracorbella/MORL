import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ══════════════════════════════════════════════════════════════════════════════
# CHANGE THIS to generate plots for a different environment
# Options: 'ADS', 'DST', 'DSTc', 'DSTm', 'RG'
ENV = 'RG'
# ══════════════════════════════════════════════════════════════════════════════

# ── Environment data ──────────────────────────────────────────────────────────
ENVS = {
    'ADS': {
        'chvi_points': np.array([
            [  0.8284,  -7.3010,  -2.4500],
            [  0.5163,  -6.7007,  -2.4500],
            [ -0.8571,  -2.9155,  -2.3188],
            [ -1.6000,  -1.2005,  -2.2331],
            [ -0.4201,  -3.4300,  -1.9355],
            [ -1.8141,  -1.2005,  -1.7150],
            [ -1.1630,  -1.7150,  -1.3352],
            [ -1.9059,   0.0000,  -1.2495],
            [ -0.2328,  -5.4392,  -1.0376],
            [ -0.3858,  -5.1709,  -0.8575],
            [ -0.8571,  -3.5037,  -0.7803],
            [ -2.2474,   0.0000,  -0.7350],
            [ -1.0100,  -3.2354,  -0.6002],
            [ -1.0100,  -4.0758,  -0.1801],
            [ -1.6000,  -1.7887,  -0.1801],
            [ -1.2285,  -3.2354,  -0.1261],
            [ -1.2285,  -3.9497,   0.0000],
            [ -1.3815,  -3.2354,   0.0000],
            [ -1.7529,  -1.5204,   0.0000],
            [ -2.0129,  -0.7143,   0.0000],
            [ -2.3223,   0.0000,   0.0000],
            [ -1.7070,  -1.6207,  -1.8411],
            [ -0.6386,  -7.0000,  -0.1261],
            [ -0.7916,  -7.0000,   0.0000],
            [  0.8284, -17.3010,  -0.7350],
            [  0.5163, -16.7007,  -0.7350],
            [  0.2041, -16.2058,  -0.1801],
            [ -0.0144, -13.6505,  -0.1261],
            [ -0.0144, -16.0798,   0.0000],
            [ -0.1673, -13.6505,   0.0000],
        ]),
        'lexhull_points': np.array([
            [-2.32,   0.00,   0.00],
            [-1.91,   0.00,  -1.25],
            [-0.01, -16.08,   0.00],
            [ 0.83, -17.30,  -0.73],
            [ 0.83,  -7.30,  -2.45],
        ]),
        'axis_labels': ('Time', 'Internal Safety', 'External Safety'),
        'output_prefix': 'ads',
    },
    'DST': {
        'chvi_points': np.array([
            [    0.7000,    -1.0000],
            [   19.7780,   -17.3831],
            [   19.0727,   -15.7057],
            [   17.8137,   -13.1254],
            [   17.3731,   -12.2479],
            [   14.8562,    -8.6483],
            [   14.0742,    -7.7255],
            [   13.1807,    -6.7935],
            [   11.0469,    -4.9010],
            [    8.0368,    -2.9701],
        ]),
        'lexhull_points': np.array([
            [    0.7000,    -1.0000],
            [   19.7780,   -17.3831],
        ]),
        'axis_labels': ('Treasure', 'Time', None),
        'output_prefix': 'dst',
    },
    'DSTc': {
        'chvi_points': np.array([
            [  103.4797,   -17.3831],
            [    1.0000,    -1.0000],
            [    2.8818,    -4.9010],
            [    4.7074,    -6.7935],
            [    7.4565,    -7.7255],
            [   21.2732,   -12.2479],
            [   63.0079,   -15.7057],
        ]),
        'lexhull_points': np.array([
            [    1.0000,    -1.0000],
            [  103.4797,   -17.3831],
        ]),
        'axis_labels': ('Treasure', 'Time', None),
        'output_prefix': 'dst_concave',
    },
    'DSTm': {
        'chvi_points': np.array([
            [  103.4797,   -17.3831],
            [    1.0000,    -1.0000],
            [    2.8818,    -4.9010],
            [    4.7074,    -6.7935],
            [    7.4565,    -7.7255],
            [   21.2732,   -12.2479],
            [   63.0079,   -15.7057],
        ]),
        'lexhull_points': np.array([
            [    1.0000,    -1.0000],
            [  103.4797,   -17.3831],
        ]),
        'axis_labels': ('Treasure', 'Time', None),
        'output_prefix': 'dst_mirrored',
    },
    'RG': {
        'chvi_points': np.array([
            [    0.0000,     0.0198,     0.0000],
            [   -0.0118,     0.0363,     0.0000],
            [   -0.0490,     0.0363,     0.0000],
            [   -0.0641,     0.0467,     0.0000],
            [   -0.0706,     0.0667,     0.0000],
            [    0.0000,     0.0023,     0.0023],
            [   -0.0706,     0.0160,     0.0160],
            [    0.0000,     0.0000,     0.0404],
            [   -0.0346,     0.0078,     0.0078],
            [   -0.0037,     0.0027,     0.0027],
        ]),
        'lexhull_points': np.array([
            [   -0.0706,     0.0667,     0.0000],
            [    0.0000,     0.0000,     0.0404],
            [    0.0000,     0.0198,     0.0000],
        ]),
        'axis_labels': ('Enemy', 'Gold', 'Gem'),
        'output_prefix': 'rg',
    },
}

# ── Load selected environment ─────────────────────────────────────────────────
cfg             = ENVS[ENV]
chvi_points     = cfg['chvi_points']
lexhull_points  = cfg['lexhull_points']
axis_labels     = cfg['axis_labels']
output_prefix   = cfg['output_prefix']
is_3d           = chvi_points.shape[1] == 3

# ── Hull computation ──────────────────────────────────────────────────────────
chvi_hull = ConvexHull(chvi_points)

# LexHull may degenerate to a line segment in 2D (< 3 points)
min_points = chvi_points.shape[1] + 1  # 3 for 2D, 4 for 3D
lexhull_hull = ConvexHull(lexhull_points) if len(lexhull_points) >= min_points else None

print(f'[{ENV}] CHVI:    {len(chvi_points)} points, {len(chvi_hull.vertices)} hull vertices')
if lexhull_hull:
    print(f'[{ENV}] LexHull: {len(lexhull_points)} points, {len(lexhull_hull.vertices)} hull vertices')
else:
    print(f'[{ENV}] LexHull: {len(lexhull_points)} points (line segment, no hull)')

# ── Colors ────────────────────────────────────────────────────────────────────
CHVI_COLOR    = '#1a78c2'
LEXHULL_COLOR = '#d84315'
ALPHA_FACE    = 0.20
GRID_COLOR    = '#cccccc'

# ── 3D helpers ────────────────────────────────────────────────────────────────
def add_hull_3d(ax, points, hull, face_color, edge_color, label):
    if hull is None:
        # 3 coplanar points: draw as a filled triangle
        poly = Poly3DCollection([points], alpha=ALPHA_FACE,
                                facecolor=face_color, edgecolor=edge_color, linewidth=0.6)
        ax.add_collection3d(poly)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   color=edge_color, s=70, edgecolors='black', linewidth=0.8,
                   zorder=6, label=label)
        return
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
        color=edge_color, s=28, zorder=5, depthshade=True, label=label,
    )
    hull_verts = points[hull.vertices]
    ax.scatter(
        hull_verts[:, 0], hull_verts[:, 1], hull_verts[:, 2],
        color=edge_color, s=70, edgecolors='black', linewidth=0.8,
        zorder=6, depthshade=False,
    )

def style_ax_3d(ax):
    ax.set_facecolor('white')
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = True
        pane.set_facecolor('#f5f5f5')
        pane.set_edgecolor('#cccccc')
    ax.tick_params(colors='black', labelsize=7)
    ax.set_xlabel(axis_labels[0], labelpad=-10)
    ax.set_ylabel(axis_labels[1], labelpad=-10)
    ax.set_zlabel(axis_labels[2], labelpad=-10)
    ax.grid(True, color=GRID_COLOR, linestyle='--', linewidth=0.4)
    ax.view_init(elev=30, azim=50)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

# ── 2D helpers ────────────────────────────────────────────────────────────────
def add_hull_2d(ax, points, hull, face_color, edge_color, label):
    if hull is None:
        # Degenerate case: just draw the line segment and its two endpoints
        ax.plot(points[:, 0], points[:, 1],
                color=edge_color, linewidth=1.2, label=label)
        ax.scatter(points[:, 0], points[:, 1],
                   color=edge_color, s=70, linewidth=0.8, zorder=6)
        return

    # Normal case: filled convex hull polygon
    hull_pts = points[hull.vertices]
    polygon  = plt.Polygon(hull_pts, closed=True,
                           facecolor=face_color, edgecolor=edge_color,
                           alpha=ALPHA_FACE, linewidth=0.8)
    ax.add_patch(polygon)
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1],
                color=edge_color, linewidth=0.8)
    ax.scatter(points[:, 0], points[:, 1],
               color=edge_color, s=28, zorder=5, label=label)
    ax.scatter(hull_pts[:, 0], hull_pts[:, 1],
               color=edge_color, s=70, edgecolors='black', linewidth=0.8, zorder=6)

def style_ax_2d(ax):
    ax.set_facecolor('#f5f5f5')
    ax.tick_params(colors='black', labelsize=7)
    ax.set_xlabel(axis_labels[0], labelpad=4)
    ax.set_ylabel(axis_labels[1], labelpad=4)
    ax.grid(True, color=GRID_COLOR, linestyle='--', linewidth=0.4)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# ── Generic wrappers that dispatch on dimensionality ──────────────────────────
def add_hull(ax, points, hull, face_color, edge_color, label):
    if is_3d:
        add_hull_3d(ax, points, hull, face_color, edge_color, label)
    else:
        add_hull_2d(ax, points, hull, face_color, edge_color, label)

def style_ax(ax):
    if is_3d:
        style_ax_3d(ax)
    else:
        style_ax_2d(ax)

def make_fig():
    fig = plt.figure(figsize=(6, 5))
    fig.patch.set_facecolor('white')
    proj = '3d' if is_3d else None
    ax   = fig.add_subplot(111, projection=proj)
    return fig, ax

# ── Plot 1: CHVI only ─────────────────────────────────────────────────────────
fig1, ax1 = make_fig()
add_hull(ax1, chvi_points, chvi_hull, CHVI_COLOR, CHVI_COLOR, 'Convex Hull')
style_ax(ax1)
plt.tight_layout(pad=2.0)
out1 = f'hull_{output_prefix}_chvi.png'
plt.savefig(out1, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved → {out1}')
plt.close()

# ── Plot 2: LexHull only ──────────────────────────────────────────────────────
fig2, ax2 = make_fig()
add_hull(ax2, lexhull_points, lexhull_hull, LEXHULL_COLOR, LEXHULL_COLOR, 'LexHull')
style_ax(ax2)
plt.tight_layout(pad=2.0)
out2 = f'hull_{output_prefix}_lexhull.png'
plt.savefig(out2, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved → {out2}')
plt.close()

# ── Plot 3: Overlay ───────────────────────────────────────────────────────────
fig3, ax3 = make_fig()
add_hull(ax3, chvi_points,    chvi_hull,    CHVI_COLOR,    CHVI_COLOR,    'Convex Hull')
add_hull(ax3, lexhull_points, lexhull_hull, LEXHULL_COLOR, LEXHULL_COLOR, 'Lexicographic Hull')
style_ax(ax3)
ax3.legend(loc='best', fontsize=8, facecolor='white', edgecolor='#cccccc')
plt.tight_layout(pad=2.0)
out3 = f'hull_{output_prefix}_overlay.png'
plt.savefig(out3, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved → {out3}')
plt.close()