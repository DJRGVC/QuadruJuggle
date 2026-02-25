"""Blender script: orient and scale target.glb, export to target.usd.

Run:
    blender --background --python scripts/blender_convert.py
"""
import math
import sys
import bpy

INPUT  = "/home/daniel-grant/Research/QuadruJuggle/assets/paddle/target.glb"
OUTPUT = "/home/daniel-grant/Research/QuadruJuggle/assets/paddle/target.usd"

# Outer ring raw X radius = 3.296 units → diameter 6.592
# Scale so outer diameter = 0.26 m
XY_SCALE = 0.26 / 6.592
# Squash to 6 mm thick
TARGET_Z = 0.006

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=INPUT)

# Rotate each mesh: face is in XZ plane → rotate -90° around X → face becomes XY (flat/horizontal)
for obj in bpy.data.objects:
    if obj.type != "MESH":
        continue
    obj.location = (0.0, 0.0, 0.0)
    obj.rotation_euler = (-math.pi / 2, 0.0, 0.0)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# Scale XY uniformly; scale Z to target thickness
for obj in bpy.data.objects:
    if obj.type != "MESH":
        continue
    zs = [v[2] for v in obj.bound_box]
    z_range = max(zs) - min(zs)
    z_scale = (TARGET_Z / z_range) if z_range > 1e-6 else 1.0
    obj.scale = (XY_SCALE, XY_SCALE, z_scale)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# Report
print("=== Final bounds ===")
for obj in bpy.data.objects:
    if obj.type != "MESH":
        continue
    xs = [v[0] for v in obj.bound_box]
    ys = [v[1] for v in obj.bound_box]
    zs = [v[2] for v in obj.bound_box]
    print(
        f"  {obj.name[:35]:35s} "
        f"X[{min(xs):+.4f},{max(xs):+.4f}] "
        f"Y[{min(ys):+.4f},{max(ys):+.4f}] "
        f"Z[{min(zs):+.4f},{max(zs):+.4f}]"
    )

bpy.ops.wm.usd_export(
    filepath=OUTPUT,
    export_materials=True,
    export_normals=True,
    root_prim_path="/target",
)
print(f"Exported: {OUTPUT}")
