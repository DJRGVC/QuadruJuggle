"""Tests for inject_ekf_reset_event helper.

Uses exec-loading to extract just the function from ball_obs_spec.py
without triggering the isaaclab import chain.
"""

import os
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Minimal fakes matching Isaac Lab's API
# ---------------------------------------------------------------------------

@dataclass
class FakeSceneEntityCfg:
    name: str = "ball"


@dataclass
class FakeEventTerm:
    func: object = None
    mode: str = ""
    params: dict = field(default_factory=dict)


def _sentinel_reset(*args, **kwargs):
    """Stand-in for reset_perception_pipeline."""
    pass


# ---------------------------------------------------------------------------
# Extract the function from source via exec (no isaaclab imports needed)
# ---------------------------------------------------------------------------

_SRC_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    "source", "go1_ball_balance", "go1_ball_balance",
    "perception", "ball_obs_spec.py",
))


def _load_inject_func():
    with open(_SRC_PATH) as f:
        src = f.read()
    marker = "def inject_ekf_reset_event("
    idx = src.index(marker)
    func_src = src[idx:]

    # The function references: EventTerm, SceneEntityCfg, reset_perception_pipeline
    ns = {
        "EventTerm": FakeEventTerm,
        "SceneEntityCfg": FakeSceneEntityCfg,
        "reset_perception_pipeline": _sentinel_reset,
    }
    exec(compile(func_src, _SRC_PATH, "exec"), ns)
    return ns["inject_ekf_reset_event"]


inject_ekf_reset_event = _load_inject_func()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeEvents:
    pass


def _make_env_cfg(events=None):
    cfg = SimpleNamespace()
    cfg.events = events or _FakeEvents()
    return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInjectEkfResetEvent:
    def test_injects_reset_event(self):
        cfg = _make_env_cfg()
        assert not hasattr(cfg.events, "reset_perception")
        inject_ekf_reset_event(cfg)
        assert hasattr(cfg.events, "reset_perception")

    def test_event_func_is_reset_perception_pipeline(self):
        cfg = _make_env_cfg()
        inject_ekf_reset_event(cfg)
        assert cfg.events.reset_perception.func is _sentinel_reset

    def test_event_mode_is_reset(self):
        cfg = _make_env_cfg()
        inject_ekf_reset_event(cfg)
        assert cfg.events.reset_perception.mode == "reset"

    def test_event_params_contain_expected_keys(self):
        cfg = _make_env_cfg()
        inject_ekf_reset_event(cfg)
        params = cfg.events.reset_perception.params
        assert "ball_cfg" in params
        assert "robot_cfg" in params
        assert "paddle_offset_b" in params

    def test_ball_cfg_is_ball(self):
        cfg = _make_env_cfg()
        inject_ekf_reset_event(cfg)
        assert cfg.events.reset_perception.params["ball_cfg"].name == "ball"

    def test_robot_cfg_is_robot(self):
        cfg = _make_env_cfg()
        inject_ekf_reset_event(cfg)
        assert cfg.events.reset_perception.params["robot_cfg"].name == "robot"

    def test_default_paddle_offset(self):
        cfg = _make_env_cfg()
        inject_ekf_reset_event(cfg)
        assert cfg.events.reset_perception.params["paddle_offset_b"] == (0.0, 0.0, 0.070)

    def test_custom_paddle_offset(self):
        cfg = _make_env_cfg()
        inject_ekf_reset_event(cfg, paddle_offset_b=(0.0, 0.0, 0.100))
        assert cfg.events.reset_perception.params["paddle_offset_b"] == (0.0, 0.0, 0.100)

    def test_idempotent_no_double_inject(self):
        cfg = _make_env_cfg()
        inject_ekf_reset_event(cfg)
        first_event = cfg.events.reset_perception
        inject_ekf_reset_event(cfg)
        assert cfg.events.reset_perception is first_event

    def test_does_not_overwrite_existing_events(self):
        events = _FakeEvents()
        events.reset_robot_joints = "existing_event"
        cfg = _make_env_cfg(events)
        inject_ekf_reset_event(cfg)
        assert cfg.events.reset_robot_joints == "existing_event"
        assert hasattr(cfg.events, "reset_perception")
