import numpy as np
from itertools import permutations

vertices_chvi = np.array([
    [-15.8655,  42.8471, -0.2380],
    [-16.5486,  44.1422,  0.0000],
    [-16.0099,  42.9283, -0.2380],
    [-11.5296,  45.0553, -0.2502],
    [-11.6814,  45.1406, -0.2502],
])

vertices_lex = np.array([
    [-16.5486,  44.1422,  0.0000],
    [-11.6814,  45.1406, -0.2502],
    [-11.5231,  44.9709, -0.2502],
])

obj_names = ["time", "delivery", "potential"]
n_obj     = 3

print("=" * 60)
print("CONVEX HULL")
print("=" * 60)

resolution = 1000
grid = [[i/resolution, j/resolution, (resolution-i-j)/resolution]
        for i in range(resolution + 1)
        for j in range(resolution + 1 - i)]
grid = np.array(grid)

best_vertex_idx  = np.argmax(grid @ vertices_chvi.T, axis=1)
winning_ch       = set()

for v_idx in range(len(vertices_chvi)):
    supporting = grid[best_vertex_idx == v_idx]
    if len(supporting) > 0:
        winning_ch.add(v_idx)
        print(f"\nVertex {v_idx}: {vertices_chvi[v_idx]}")
        print(f"  Supported by {len(supporting)} weight vectors")
        print(f"  Centroid: {supporting.mean(axis=0).round(4)}")

print(f"\nOptimal convex hull vertices ({len(winning_ch)}):")
print(f"  [   time   ,  delivery , potential ]   [  w_time  , w_delivery, w_potential]")
for v_idx in sorted(winning_ch):
    v        = vertices_chvi[v_idx]
    supporting = grid[best_vertex_idx == v_idx]
    centroid = supporting.mean(axis=0) # * resolution
    print(f"  [{v[0]:10.4f}, {v[1]:10.4f}, {v[2]:10.4f}]   [{centroid[0]:10.4f}, {centroid[1]:10.4f}, {centroid[2]:10.4f}]")

print("\n" + "=" * 60)
print("CONVEX HULL — canonical priority weights")
print("=" * 60)

canon_weights = np.array([
    [1,     100,   10000],
    [1,     10000, 100  ],
    [100,   1,     10000],
    [100,   10000, 1    ],
    [10000, 1,     100  ],
    [10000, 100,   1    ],
])

for w in canon_weights:
    best_idx = np.argmax(vertices_chvi @ w)
    print(f"Weights {w} -> Vertex {best_idx}: {vertices_chvi[best_idx]}")

print("\n" + "=" * 60)
print("LEXICOGRAPHIC HULL")
print("=" * 60)

winning_lex = set()
for perm in permutations(range(n_obj)):
    scores   = vertices_lex[:, perm]
    best_idx = np.lexsort(scores.T[::-1])[-1]
    winning_lex.add(best_idx)

    canon_w = np.zeros(n_obj)
    for rank, obj in enumerate(perm):
        canon_w[obj] = [10000, 100, 1][rank]

    print(f"\nPriority {list(perm)} -> Vertex {best_idx}: {vertices_lex[best_idx]}")
    print(f"  Canonical weight vector: {canon_w}")

print(f"\nOptimal lex hull vertices ({len(winning_lex)}):")
print(f"  [   time   ,  delivery , potential ]")
for v in vertices_lex[sorted(winning_lex)]:
    print(f"  [{v[0]:10.4f}, {v[1]:10.4f}, {v[2]:10.4f}]")