#!/usr/bin/env python3
"""Generate assets/paddle/disc.usda — a 64-sided flat disc with physics APIs baked in.

Run once before first training / preview:
    python scripts/create_disc_usd.py

No external dependencies (pure Python + math).
"""

import math
import os

# ---------------------------------------------------------------------------
# Geometry parameters  (keep in sync with ball_balance_env_cfg.py constants)
# ---------------------------------------------------------------------------
NUM_SIDES  = 64
RADIUS     = 0.085    # metres  (170 mm diameter / 2)
THICKNESS  = 0.007    # metres  (7 mm)
MASS_KG    = 0.15

# ---------------------------------------------------------------------------
# Build vertex array
# ---------------------------------------------------------------------------
half_h = THICKNESS / 2.0

# Index layout:
#  0              : top centre
#  1 … N          : top ring (CCW from above)
#  N+1            : bottom centre
#  N+2 … 2N+1     : bottom ring
def ring(z):
    return [
        (RADIUS * math.cos(2 * math.pi * i / NUM_SIDES),
         RADIUS * math.sin(2 * math.pi * i / NUM_SIDES),
         z)
        for i in range(NUM_SIDES)
    ]

N  = NUM_SIDES
TC = 0
BC = N + 1
top_ring    = ring(+half_h)
bottom_ring = ring(-half_h)
points = [(0.0, 0.0, +half_h)] + top_ring + [(0.0, 0.0, -half_h)] + bottom_ring

# ---------------------------------------------------------------------------
# Build face lists
# ---------------------------------------------------------------------------
fvc   = []   # faceVertexCounts
fvi   = []   # faceVertexIndices

# Top cap
for i in range(N):
    j = (i + 1) % N
    fvc.append(3)
    fvi += [TC, 1 + i, 1 + j]

# Bottom cap (reversed winding for outward normals)
for i in range(N):
    j = (i + 1) % N
    fvc.append(3)
    fvi += [BC, N + 2 + j, N + 2 + i]

# Side quads → 2 triangles each
for i in range(N):
    j = (i + 1) % N
    t0, t1 = 1 + i,     1 + j
    b0, b1 = N + 2 + i, N + 2 + j
    fvc += [3, 3]
    fvi += [t0, b0, t1,
            t1, b0, b1]

# ---------------------------------------------------------------------------
# Serialise to USD ASCII
# ---------------------------------------------------------------------------
def pts_str(pts):
    return ", ".join(f"({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f})" for p in pts)

def ints_str(lst, per_line=16):
    rows = []
    for start in range(0, len(lst), per_line):
        rows.append("        " + ", ".join(str(x) for x in lst[start:start + per_line]))
    return ",\n".join(rows)

usda = f"""#usda 1.0
(
    upAxis = "Z"
    metersPerUnit = 1
    defaultPrim = "Paddle"
)

# ---------------------------------------------------------------------------
# Paddle disc: {N}-sided cylinder, radius {RADIUS} m, thickness {THICKNESS*1000:.0f} mm
# ---------------------------------------------------------------------------
def Xform "Paddle" (
    prepend apiSchemas = [
        "PhysicsRigidBodyAPI",
        "PhysxRigidBodyAPI",
        "PhysicsMassAPI"
    ]
)
{{
    bool physics:rigidBodyEnabled  = true
    bool physics:kinematicEnabled  = true
    bool physxRigidBody:disableGravity = true
    float physics:mass             = {MASS_KG}

    def Mesh "disc_mesh" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {{
        uniform token subdivisionScheme = "none"

        # {len(points)} vertices
        point3f[] points = [{pts_str(points)}]

        # {len(fvc)} faces  ({len(fvc) * 3} indices)
        int[] faceVertexCounts = [{", ".join(str(x) for x in fvc)}]
        int[] faceVertexIndices = [
{ints_str(fvi)}
        ]

        # Collision approximation
        token physics:approximation = "convexHull"

        # Orange visual
        color3f[] primvars:displayColor = [(1.0, 0.35, 0.0)]
    }}
}}
"""

out_path = os.path.join(os.path.dirname(__file__), "..", "assets", "paddle", "disc.usda")
out_path = os.path.normpath(out_path)
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "w") as f:
    f.write(usda)

print(f"Written: {out_path}")
print(f"  {N} sides, radius={RADIUS} m, thickness={THICKNESS*1000:.0f} mm, mass={MASS_KG} kg")
print(f"  {len(points)} vertices, {len(fvc)} triangles")
