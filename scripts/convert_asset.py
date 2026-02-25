"""Convert a GLB/FBX/OBJ asset to USD using Isaac Sim's built-in converter.

Run from the IsaacLab directory so Isaac Sim initialises correctly:
    cd ~/IsaacLab
    python ~/Research/QuadruJuggle/scripts/convert_asset.py

Output: assets/paddle/target.usd  (next to the source GLB)
"""

import argparse
import asyncio
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert GLB → USD via Isaac Sim asset converter.")
parser.add_argument(
    "--input",
    default=os.path.expanduser("~/Research/QuadruJuggle/assets/paddle/target.glb"),
    help="Path to source asset (GLB/FBX/OBJ).",
)
parser.add_argument(
    "--output",
    default=os.path.expanduser("~/Research/QuadruJuggle/assets/paddle/target.usd"),
    help="Path for converted USD output.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Force headless — no GUI needed for conversion
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.kit.asset_converter as converter


async def convert(input_path: str, output_path: str) -> None:
    context = converter.AssetConverterContext()
    # Keep textures embedded in USD
    context.embed_textures = True
    # Import normals from source
    context.import_normals = True

    task = converter.get_instance().create_converter_task(input_path, output_path, None, context)
    success = await task.wait_until_finished()
    if not success:
        print(f"[ERROR] Conversion failed: {task.get_error_message()}")
        sys.exit(1)
    print(f"[OK] Converted: {output_path}")


asyncio.get_event_loop().run_until_complete(convert(args_cli.input, args_cli.output))

print()
print("Next steps:")
print("  1. Open the USD in Isaac Sim to inspect it")
print("  2. Add a collision mesh (Physics > Add > Collision > Convex Decomposition)")
print("     or use a simple disc/box approximation for the collider")
print("  3. The env cfg will then reference: assets/paddle/target.usd")

simulation_app.close()
