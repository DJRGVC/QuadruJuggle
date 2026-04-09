"""Depth-frame visualizer for camera feed panel in teleop UI.

Renders colorized depth images with detection bbox overlay, ball marker,
and optional telemetry text. Designed to be embedded as a panel in
play_teleop_ui.py or any OpenCV-based visualization.

Usage::

    viz = DepthFrameVisualizer(width=320, height=240)

    # From uint16 depth (mm) — real D435i frames
    panel = viz.render(depth_u16_mm, detection=det)

    # From float32 depth (m) — Isaac Lab TiledCamera
    panel = viz.render_f32(depth_f32_m, sim_detection=sim_det)

    # Compose into larger UI
    full_ui = np.hstack([existing_panel, panel])
    cv2.imshow("Teleop", full_ui)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

try:
    import cv2  # type: ignore[import-untyped]
except ImportError:
    cv2 = None

if TYPE_CHECKING:
    from ..real.detector import Detection
    from ..sim_detector import SimDetection


@dataclass
class VizConfig:
    """Configuration for DepthFrameVisualizer."""

    width: int = 320  # output panel width
    height: int = 240  # output panel height
    min_depth_mm: int = 100  # colormap floor (mm)
    max_depth_mm: int = 1500  # colormap ceiling (mm)
    colormap: int = 2  # cv2.COLORMAP_JET = 2
    bbox_color: tuple[int, int, int] = (0, 255, 0)  # green
    bbox_thickness: int = 2
    marker_color: tuple[int, int, int] = (0, 0, 255)  # red circle
    marker_radius: int = 6
    text_color: tuple[int, int, int] = (255, 255, 255)
    text_bg_color: tuple[int, int, int] = (0, 0, 0)
    font_scale: float = 0.45
    show_telemetry: bool = True
    title: str = "Depth Camera"


class DepthFrameVisualizer:
    """Renders colorized depth frames with detection overlays.

    Accepts either uint16 depth in mm (real D435i) or float32 depth in
    metres (Isaac Lab TiledCamera). Outputs a BGR image suitable for
    cv2.imshow or compositing into a larger UI.

    Args:
        cfg: Visualization config. Uses defaults if not provided.
    """

    def __init__(self, cfg: VizConfig | None = None) -> None:
        if cv2 is None:
            raise ImportError("cv2 is required for DepthFrameVisualizer")
        self._cfg = cfg or VizConfig()
        self._frame_count = 0

    def render(
        self,
        depth_u16_mm: np.ndarray,
        detection: Detection | None = None,
        telemetry: dict[str, str] | None = None,
    ) -> np.ndarray:
        """Render uint16 depth frame (mm) with detection overlay.

        Args:
            depth_u16_mm: (H, W) uint16 depth in millimetres.
            detection: Optional Detection with bbox and pos_cam.
            telemetry: Optional dict of key-value strings to overlay.

        Returns:
            (cfg.height, cfg.width, 3) BGR uint8 image.
        """
        self._frame_count += 1
        cfg = self._cfg

        # Colorize depth
        vis = self._colorize_u16(depth_u16_mm)

        # Resize to output dimensions
        h_in, w_in = vis.shape[:2]
        vis = cv2.resize(vis, (cfg.width, cfg.height), interpolation=cv2.INTER_LINEAR)
        sx = cfg.width / w_in
        sy = cfg.height / h_in

        # Draw detection overlay
        if detection is not None:
            x1, y1, x2, y2 = detection.bbox
            # Scale bbox to output size
            bx1 = int(x1 * sx)
            by1 = int(y1 * sy)
            bx2 = int(x2 * sx)
            by2 = int(y2 * sy)
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), cfg.bbox_color, cfg.bbox_thickness)

            # Centre marker
            cx = (bx1 + bx2) // 2
            cy = (by1 + by2) // 2
            cv2.circle(vis, (cx, cy), cfg.marker_radius, cfg.marker_color, -1)

            # Depth + confidence label
            depth_m = detection.pos_cam[2] if hasattr(detection, 'pos_cam') else 0
            conf = detection.confidence if hasattr(detection, 'confidence') else 0
            label = f"{depth_m:.2f}m ({conf:.0%})"
            self._draw_label(vis, label, (bx1, by1 - 5))

        # Title bar
        self._draw_title(vis, cfg.title)

        # Telemetry text
        if cfg.show_telemetry and telemetry:
            self._draw_telemetry(vis, telemetry)

        # "No detection" indicator
        if detection is None:
            self._draw_label(vis, "NO DETECTION", (cfg.width // 2 - 50, cfg.height // 2),
                             color=(0, 0, 255))

        return vis

    def render_f32(
        self,
        depth_f32_m: np.ndarray,
        sim_detection: SimDetection | None = None,
        telemetry: dict[str, str] | None = None,
    ) -> np.ndarray:
        """Render float32 depth frame (metres) with sim detection overlay.

        Converts to uint16 mm internally, then wraps SimDetection into
        the same format as Detection for rendering.

        Args:
            depth_f32_m: (H, W) float32 depth in metres.
            sim_detection: Optional SimDetection from SimBallDetector.
            telemetry: Optional dict of key-value strings to overlay.

        Returns:
            (cfg.height, cfg.width, 3) BGR uint8 image.
        """
        # Convert float32 metres to uint16 mm
        valid = np.isfinite(depth_f32_m) & (depth_f32_m > 0)
        depth_mm = np.zeros(depth_f32_m.shape, dtype=np.uint16)
        depth_mm[valid] = np.clip(depth_f32_m[valid] * 1000, 0, 65535).astype(np.uint16)

        # Wrap SimDetection as a Detection-like object for render()
        det_wrapper = None
        if sim_detection is not None:
            det_wrapper = _SimDetAsDetection(sim_detection)

        return self.render(depth_mm, detection=det_wrapper, telemetry=telemetry)

    def _colorize_u16(self, depth_mm: np.ndarray) -> np.ndarray:
        """Colorize uint16 depth with COLORMAP_JET."""
        cfg = self._cfg
        d = depth_mm.astype(np.float32)
        valid = (depth_mm >= cfg.min_depth_mm) & (depth_mm <= cfg.max_depth_mm)

        # Normalize to 0-255
        norm = np.zeros(d.shape, dtype=np.uint8)
        if valid.any():
            norm[valid] = np.clip(
                (d[valid] - cfg.min_depth_mm) * 255.0 / (cfg.max_depth_mm - cfg.min_depth_mm),
                0, 255,
            ).astype(np.uint8)

        # Apply colormap (invalid pixels stay black)
        colored = cv2.applyColorMap(norm, cfg.colormap)
        colored[~valid] = 0
        return colored

    def _draw_title(self, img: np.ndarray, text: str) -> None:
        """Draw title bar at top of image."""
        cfg = self._cfg
        h = int(20 * cfg.font_scale / 0.45)
        cv2.rectangle(img, (0, 0), (cfg.width, h), cfg.text_bg_color, -1)
        cv2.putText(
            img, text, (4, h - 4),
            cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, cfg.text_color, 1,
        )

    def _draw_label(
        self, img: np.ndarray, text: str, pos: tuple[int, int],
        color: tuple[int, int, int] | None = None,
    ) -> None:
        """Draw text with background at given position."""
        cfg = self._cfg
        c = color or cfg.text_color
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, 1)
        x, y = pos
        cv2.rectangle(img, (x, y - th - 2), (x + tw + 2, y + 2), cfg.text_bg_color, -1)
        cv2.putText(img, text, (x + 1, y), cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, c, 1)

    def _draw_telemetry(self, img: np.ndarray, telemetry: dict[str, str]) -> None:
        """Draw key-value telemetry text at bottom of image."""
        cfg = self._cfg
        line_h = int(16 * cfg.font_scale / 0.45)
        y = cfg.height - 4
        for key, val in reversed(list(telemetry.items())):
            text = f"{key}: {val}"
            cv2.putText(
                img, text, (4, y),
                cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale * 0.85, cfg.text_color, 1,
            )
            y -= line_h


class _SimDetAsDetection:
    """Adapter wrapping SimDetection to look like Detection for rendering."""

    def __init__(self, sim_det: SimDetection) -> None:
        self.pos_cam = sim_det.pos_cam
        self.confidence = 1.0  # sim detections are always "confident"
        # Estimate bbox from pixel centre + blob size
        u, v = sim_det.pixel_uv
        r = max(5, int(np.sqrt(sim_det.num_pixels / np.pi)))
        self.bbox = (int(u - r), int(v - r), int(u + r), int(v + r))
        self.method = "sim"
