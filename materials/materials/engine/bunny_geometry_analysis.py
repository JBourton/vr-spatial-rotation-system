import numpy as np
import math

# ---- load and normalise exactly as the coursework engine does ----
V, Vi = [], []
with open("bunny.obj") as f:
    for line in f:
        vals = line.split()
        if not vals:
            continue
        if vals[0] == 'v':
            V.append([float(x) for x in vals[1:4]])
        elif vals[0] == 'f':
            Vi.append([int(x) for x in vals[1:4]])

V = np.array(V)
Vi = np.array(Vi) - 1

vmin, vmax = V.min(), V.max()
V = (2 * (V - vmin) / (vmax - vmin) - 1) * 1.25
V[:, 0] -= (V[:, 0].min() + V[:, 0].max()) / 2
V[:, 1] -= (V[:, 1].min() + V[:, 1].max()) / 2

print("=" * 55)
print("BUNNY MESH STATISTICS (normalised, as rendered)")
print("=" * 55)
print(f"Vertices : {len(V)}")
print(f"Faces    : {len(Vi)}")
print()

# ---- axis extents ----
xmin, xmax = V[:,0].min(), V[:,0].max()
ymin, ymax = V[:,1].min(), V[:,1].max()
zmin, zmax = V[:,2].min(), V[:,2].max()
print(f"X extent : {xmin:.3f} to {xmax:.3f}  (width  = {xmax-xmin:.3f})")
print(f"Y extent : {ymin:.3f} to {ymax:.3f}  (height = {ymax-ymin:.3f})")
print(f"Z extent : {zmin:.3f} to {zmax:.3f}  (depth  = {zmax-zmin:.3f})")
print()

# ---- bounding sphere from centroid ----
centroid = V.mean(axis=0)
dists = np.linalg.norm(V - centroid, axis=1)
bs_radius = dists.max()
bs_volume = (4/3) * math.pi * bs_radius**3
print(f"Centroid              : ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})")
print(f"Bounding sphere radius: {bs_radius:.4f}")
print(f"Bounding sphere volume: {bs_volume:.4f}")
print()

# ---- AABB ----
half = (V.max(axis=0) - V.min(axis=0)) / 2
aabb_volume = np.prod(2 * half)
print(f"AABB half-extents     : x={half[0]:.4f}  y={half[1]:.4f}  z={half[2]:.4f}")
print(f"AABB volume           : {aabb_volume:.4f}")
print()

# ---- volumes used in your simulation ----
r_boss   = 0.20
r_teeny  = 0.12
v_boss   = (4/3)*math.pi*r_boss**3
v_teeny  = (4/3)*math.pi*r_teeny**3
print(f"Sim sphere r=0.20 volume : {v_boss:.4f}")
print(f"Sim sphere r=0.12 volume : {v_teeny:.4f}")
print()

# ---- how much empty space do the sim spheres waste? ----
from scipy.spatial import ConvexHull
hull = ConvexHull(V)
print(f"Convex hull volume    : {hull.volume:.4f}")
print(f"Convex hull faces     : {len(hull.simplices)}")
print()

print("=" * 55)
print("VOLUME EFFICIENCY (what fraction is actually bunny)")
print("=" * 55)
print(f"Convex hull / bounding sphere (r=bs_radius) : {100*hull.volume/bs_volume:.1f}%")
print(f"Convex hull / AABB                          : {100*hull.volume/aabb_volume:.1f}%")
print(f"Convex hull / sim sphere r=0.20             : {100*hull.volume/v_boss:.1f}%")
print(f"AABB        / sim sphere r=0.20             : {100*aabb_volume/v_boss:.1f}%")
print()

print("=" * 55)
print("ASPECT RATIO (how non-spherical is the bunny?)")
print("=" * 55)
height = ymax - ymin
width  = xmax - xmin
depth  = zmax - zmin
print(f"Height/Width : {height/width:.2f}  (>1 means taller than wide)")
print(f"Height/Depth : {height/depth:.2f}")
print()

print("=" * 55)
print("COLLISION RADIUS SUITABILITY")
print("=" * 55)
# At r=0.20, what fraction of the model sticks OUT of the sphere?
vertices_outside_02 = np.sum(np.linalg.norm(V - centroid, axis=1) > 0.20)
vertices_outside_12 = np.sum(np.linalg.norm(V - centroid, axis=1) > 0.12)
print(f"Vertices outside r=0.20 sphere: {vertices_outside_02} / {len(V)} ({100*vertices_outside_02/len(V):.1f}%)")
print(f"Vertices outside r=0.12 sphere: {vertices_outside_12} / {len(V)} ({100*vertices_outside_12/len(V):.1f}%)")
print()
print(f"Min radius to contain ALL vertices: {bs_radius:.4f}")
print(f"(r=0.20 {'DOES' if bs_radius<=0.20 else 'does NOT'} contain the full model)")
