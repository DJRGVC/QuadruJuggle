"""UserVelocityInput — reads joystick or keyboard and returns normalised vx/vy.

Supports three backends:
  - "pygame"   : USB gamepad (left stick axes 0/1). Default.
  - "keyboard" : WASD keyboard fallback (requires pynput).
  - "zero"     : returns (0, 0) — for testing without hardware.

Physical units: m/s. Output normalised to [-1, +1] for pi2 command.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Literal

import torch

# Pi2 velocity scaling factors (from action_term._CMD_SCALES indices 6,7)
_VX_SCALE = 0.5  # m/s
_VY_SCALE = 0.5  # m/s

# Safety clamp: restrict user velocity to a conservative fraction of pi2 max
_USER_VX_MAX = 0.30  # m/s  (60% of training max)
_USER_VY_MAX = 0.30  # m/s


@dataclass
class UserVelocityInputCfg:
    backend: Literal["pygame", "keyboard", "zero"] = "pygame"
    gamepad_axis_fwd: int = 1   # left stick vertical
    gamepad_axis_lat: int = 0   # left stick horizontal
    invert_fwd: bool = True     # most gamepads have inverted Y
    invert_lat: bool = False
    max_vx: float = _USER_VX_MAX
    max_vy: float = _USER_VY_MAX
    deadband: float = 0.05      # fraction of max axis value


class UserVelocityInput:
    """Thread-safe joystick/keyboard reader returning (vx, vy) in m/s."""

    def __init__(self, cfg: UserVelocityInputCfg):
        self.cfg = cfg
        self._vx = 0.0
        self._vy = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.cfg.backend == "zero":
            return
        if self.cfg.backend == "pygame":
            self._init_pygame()
            self._thread = threading.Thread(target=self._pygame_loop, daemon=True)
        elif self.cfg.backend == "keyboard":
            self._init_keyboard()
            self._thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self._running = True
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def get_cmd(self) -> tuple[float, float]:
        """Return (vx, vy) in m/s with safety clamp."""
        with self._lock:
            return (self._vx, self._vy)

    def get_cmd_normalized(self) -> tuple[float, float]:
        """Return (vx_norm, vy_norm) in [-1, +1] for pi2 command indices 6,7."""
        vx, vy = self.get_cmd()
        return (vx / _VX_SCALE, vy / _VY_SCALE)

    def get_cmd_tensor(self, num_envs: int, device: str) -> torch.Tensor:
        """Return (num_envs, 2) tensor [vx_norm, vy_norm] broadcast to all envs."""
        vx_n, vy_n = self.get_cmd_normalized()
        return torch.tensor([[vx_n, vy_n]], device=device).expand(num_envs, -1)

    # -- pygame backend ----------------------------------------------------

    def _init_pygame(self) -> None:
        import pygame
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError(
                "No joystick detected. Use backend='keyboard' or backend='zero'."
            )
        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()
        print(f"[UserVelocityInput] joystick: {self._joystick.get_name()}")

    def _pygame_loop(self) -> None:
        import pygame
        cfg = self.cfg
        while self._running:
            pygame.event.pump()
            raw_fwd = self._joystick.get_axis(cfg.gamepad_axis_fwd)
            raw_lat = self._joystick.get_axis(cfg.gamepad_axis_lat)
            if cfg.invert_fwd:
                raw_fwd = -raw_fwd
            if cfg.invert_lat:
                raw_lat = -raw_lat
            if abs(raw_fwd) < cfg.deadband:
                raw_fwd = 0.0
            if abs(raw_lat) < cfg.deadband:
                raw_lat = 0.0
            vx = max(-cfg.max_vx, min(cfg.max_vx, raw_fwd * cfg.max_vx))
            vy = max(-cfg.max_vy, min(cfg.max_vy, raw_lat * cfg.max_vy))
            with self._lock:
                self._vx = float(vx)
                self._vy = float(vy)
            time.sleep(0.01)

    # -- keyboard backend (WASD) -------------------------------------------

    def _init_keyboard(self) -> None:
        try:
            from pynput import keyboard  # noqa: F401
        except ImportError:
            raise ImportError("pip install pynput")
        self._keys_held: set[str] = set()

    def _keyboard_loop(self) -> None:
        from pynput import keyboard as pynput_kb

        def on_press(key):
            try:
                self._keys_held.add(key.char)
            except AttributeError:
                pass

        def on_release(key):
            try:
                self._keys_held.discard(key.char)
            except AttributeError:
                pass

        listener = pynput_kb.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        cfg = self.cfg
        while self._running:
            vx, vy = 0.0, 0.0
            if "w" in self._keys_held:
                vx += cfg.max_vx
            if "s" in self._keys_held:
                vx -= cfg.max_vx
            if "a" in self._keys_held:
                vy += cfg.max_vy
            if "d" in self._keys_held:
                vy -= cfg.max_vy
            with self._lock:
                self._vx = vx
                self._vy = vy
            time.sleep(0.01)
        listener.stop()
