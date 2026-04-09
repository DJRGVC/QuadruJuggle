"""Generate synthetic depth-image training data for YOLO ball detection.

Renders a 40mm ping-pong ball at random positions in the D435i depth field
of view, with realistic depth noise, and outputs YOLO-format annotations.

Output structure:
    <out_dir>/
        images/     # 16-bit PNG depth frames (uint16, millimetres)
        labels/     # YOLO txt files: class cx cy w h (normalised)

Usage:
    python scripts/perception/generate_yolo_data.py --out data/yolo_ball --n 5000

Each image contains exactly one ball rendered as a filled disc in a depth
frame, with D435i-like depth noise applied to the entire frame and
distance-dependent noise on the ball region.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

try:
    import cv2  # type: ignore[import-untyped]
except ImportError:
    cv2 = None


# D435i 848x480 depth intrinsics (typical factory values)
_DEFAULT_FX = 425.19
_DEFAULT_FY = 425.19
_DEFAULT_CX = 423.36
_DEFAULT_CY = 239.95
_DEFAULT_W = 848
_DEFAULT_H = 480

# Ball parameters
_BALL_RADIUS_M = 0.020  # 40mm ping-pong ball

# Depth range (metres)
_MIN_DEPTH = 0.168  # D435i minimum
_MAX_DEPTH = 1.5    # practical limit for 40mm ball


def render_ball_depth_frame(
    ball_x: float,
    ball_y: float,
    ball_z: float,
    fx: float = _DEFAULT_FX,
    fy: float = _DEFAULT_FY,
    cx: float = _DEFAULT_CX,
    cy: float = _DEFAULT_CY,
    width: int = _DEFAULT_W,
    height: int = _DEFAULT_H,
    ball_radius: float = _BALL_RADIUS_M,
    bg_depth_mean: float = 3.0,
    bg_depth_std: float = 0.5,
    noise_sigma_base: float = 1.0,
    noise_sigma_quad: float = 5.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """Render a depth frame with a single ball.

    Args:
        ball_x, ball_y, ball_z: Ball centre in camera optical frame (metres).
        fx, fy, cx, cy: Intrinsics.
        width, height: Frame dimensions.
        ball_radius: Ball radius in metres.
        bg_depth_mean: Background depth mean (metres). Set high to simulate
            upward-facing camera seeing sky/ceiling.
        bg_depth_std: Background depth variation.
        noise_sigma_base: Base depth noise (mm).
        noise_sigma_quad: Quadratic depth noise coefficient (mm/m²).
        rng: NumPy random generator.

    Returns:
        depth_u16: (H, W) uint16 depth frame in millimetres.
        bbox: (x1, y1, x2, y2) pixel bounding box of the ball, or None
              if ball projects outside frame.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Background: mostly far away or invalid (0)
    # Simulate upward-facing camera: background is sky/ceiling at variable depth
    # 30% of pixels are invalid (no return from sky)
    bg_depth_m = rng.normal(bg_depth_mean, bg_depth_std, size=(height, width))
    bg_depth_m = np.clip(bg_depth_m, 0.5, 10.0)
    invalid_mask = rng.random((height, width)) < 0.30
    bg_depth_mm = (bg_depth_m * 1000).astype(np.float64)
    bg_depth_mm[invalid_mask] = 0  # invalid depth

    # Project ball centre to pixel coordinates
    u_c = fx * ball_x / ball_z + cx
    v_c = fy * ball_y / ball_z + cy

    # Ball apparent radius in pixels
    r_px = fx * ball_radius / ball_z

    # Bounding box
    x1 = int(np.floor(u_c - r_px))
    y1 = int(np.floor(v_c - r_px))
    x2 = int(np.ceil(u_c + r_px)) + 1
    y2 = int(np.ceil(v_c + r_px)) + 1

    # Check if ball is within frame (at least partially)
    if x2 <= 0 or x1 >= width or y2 <= 0 or y1 >= height:
        return bg_depth_mm.astype(np.uint16), None

    # Clamp bbox to frame
    bx1 = max(0, x1)
    by1 = max(0, y1)
    bx2 = min(width, x2)
    by2 = min(height, y2)

    # Render ball as a sphere: each pixel gets depth from the sphere surface
    # For pixel (u, v), ray direction is ((u-cx)/fx, (v-cy)/fy, 1)
    # Intersect with sphere at (ball_x, ball_y, ball_z) with radius ball_radius
    uu = np.arange(bx1, bx2, dtype=np.float64)
    vv = np.arange(by1, by2, dtype=np.float64)
    uu_grid, vv_grid = np.meshgrid(uu, vv)

    # Ray directions (not normalised — we solve parametric t along the ray)
    dx = (uu_grid - cx) / fx
    dy = (vv_grid - cy) / fy
    dz = np.ones_like(dx)

    # Solve |origin + t*dir - centre|² = r² for t
    # origin = (0,0,0), centre = (ball_x, ball_y, ball_z)
    # a = dx²+dy²+dz², b = -2(dx*bx + dy*by + dz*bz), c = bx²+by²+bz² - r²
    a = dx * dx + dy * dy + dz * dz
    b = -2.0 * (dx * ball_x + dy * ball_y + dz * ball_z)
    c = ball_x**2 + ball_y**2 + ball_z**2 - ball_radius**2

    discriminant = b * b - 4 * a * c
    hit = discriminant >= 0

    # t for front intersection
    t_front = np.where(hit, (-b - np.sqrt(np.maximum(discriminant, 0))) / (2 * a), 0)
    sphere_depth_m = t_front * dz  # depth = t * dz (z-component)

    # Only draw pixels where ray hits the sphere
    ball_mask = hit & (sphere_depth_m > _MIN_DEPTH)

    # Apply ball depth with per-pixel noise
    ball_depth_mm = sphere_depth_m * 1000  # metres to mm
    z_for_noise = sphere_depth_m
    sigma_mm = noise_sigma_base + noise_sigma_quad * z_for_noise * z_for_noise
    ball_noise = rng.normal(0, 1, size=ball_depth_mm.shape) * sigma_mm
    ball_depth_mm += ball_noise

    # Stamp ball into background
    depth_frame = bg_depth_mm.copy()
    patch = depth_frame[by1:by2, bx1:bx2]
    patch[ball_mask] = ball_depth_mm[ball_mask]
    depth_frame[by1:by2, bx1:bx2] = patch

    # Add global depth noise (D435i temporal noise)
    valid = depth_frame > 0
    global_z_m = depth_frame / 1000.0
    global_sigma = noise_sigma_base + noise_sigma_quad * global_z_m * global_z_m
    global_noise = rng.normal(0, 1, size=depth_frame.shape) * global_sigma
    depth_frame[valid] += global_noise[valid]

    # Clamp and convert to uint16
    depth_frame = np.clip(depth_frame, 0, 65535).astype(np.uint16)

    # Tight bbox from actual ball pixels
    if not np.any(ball_mask):
        return depth_frame, None

    ball_ys, ball_xs = np.where(ball_mask)
    tight_x1 = int(bx1 + ball_xs.min())
    tight_y1 = int(by1 + ball_ys.min())
    tight_x2 = int(bx1 + ball_xs.max()) + 1
    tight_y2 = int(by1 + ball_ys.max()) + 1

    return depth_frame, (tight_x1, tight_y1, tight_x2, tight_y2)


def bbox_to_yolo(
    bbox: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    class_id: int = 0,
) -> str:
    """Convert pixel bbox to YOLO format: class cx cy w h (normalised)."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def generate_dataset(
    out_dir: str,
    n_images: int = 5000,
    seed: int = 42,
    width: int = _DEFAULT_W,
    height: int = _DEFAULT_H,
    fx: float = _DEFAULT_FX,
    fy: float = _DEFAULT_FY,
    cx: float = _DEFAULT_CX,
    cy: float = _DEFAULT_CY,
) -> dict[str, int]:
    """Generate a synthetic YOLO dataset.

    Returns:
        Stats dict with keys: total, valid, skipped.
    """
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    img_dir = out / "images"
    lbl_dir = out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    if cv2 is None:
        raise ImportError("cv2 required for PNG writing. pip install opencv-python")

    stats = {"total": n_images, "valid": 0, "skipped": 0}

    for i in range(n_images):
        # Sample ball position in camera frame
        # z: uniform in [0.20, 1.5] metres (operational range)
        z = rng.uniform(0.20, _MAX_DEPTH)

        # x, y: sample within the camera FOV at this depth
        # FOV bounds: x ∈ [-cx/fx * z, (w-cx)/fx * z], same for y
        x_min = -(cx - 30) / fx * z  # margin of 30px from edge
        x_max = (width - cx - 30) / fx * z
        y_min = -(cy - 30) / fy * z
        y_max = (height - cy - 30) / fy * z
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)

        depth_frame, bbox = render_ball_depth_frame(
            ball_x=x, ball_y=y, ball_z=z,
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=width, height=height,
            rng=rng,
        )

        if bbox is None:
            stats["skipped"] += 1
            continue

        # Check minimum ball size (must be at least 3px radius to be detectable)
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        if bw < 4 or bh < 4:
            stats["skipped"] += 1
            continue

        fname = f"{i:06d}"
        cv2.imwrite(str(img_dir / f"{fname}.png"), depth_frame)

        yolo_line = bbox_to_yolo(bbox, width, height)
        (lbl_dir / f"{fname}.txt").write_text(yolo_line + "\n")
        stats["valid"] += 1

    # Write dataset.yaml for ultralytics
    yaml_content = (
        f"path: {out.resolve()}\n"
        f"train: images\n"
        f"val: images\n"
        f"nc: 1\n"
        f"names: ['ball']\n"
    )
    (out / "dataset.yaml").write_text(yaml_content)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic YOLO ball training data")
    parser.add_argument("--out", type=str, default="data/yolo_ball", help="Output directory")
    parser.add_argument("--n", type=int, default=5000, help="Number of images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    stats = generate_dataset(args.out, n_images=args.n, seed=args.seed)
    print(f"Generated {stats['valid']} images ({stats['skipped']} skipped)")
    print(f"Output: {args.out}/")


if __name__ == "__main__":
    main()
