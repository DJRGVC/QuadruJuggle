"""Tests for velocity command modules (UserVelocityInput + CommandMixer).

All tests use the 'zero' backend — no hardware required.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest

import torch

# Direct imports — bypass go1_ball_balance/__init__ (pulls Isaac Lab/pxr).
_VEL_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "source", "go1_ball_balance", "go1_ball_balance", "vel_cmd",
))


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_mixer_mod = _load("command_mixer", os.path.join(_VEL_DIR, "command_mixer.py"))
_uvi_mod = _load("user_velocity_input", os.path.join(_VEL_DIR, "user_velocity_input.py"))

CommandMixer = _mixer_mod.CommandMixer
CommandMixerCfg = _mixer_mod.CommandMixerCfg
UserVelocityInput = _uvi_mod.UserVelocityInput
UserVelocityInputCfg = _uvi_mod.UserVelocityInputCfg
_VX_SCALE = _uvi_mod._VX_SCALE
_VY_SCALE = _uvi_mod._VY_SCALE


# -----------------------------------------------------------------------
# UserVelocityInput tests (zero backend)
# -----------------------------------------------------------------------

class TestUserVelocityInputZero(unittest.TestCase):
    """Tests with the 'zero' backend (no hardware)."""

    def setUp(self):
        self.uvi = UserVelocityInput(UserVelocityInputCfg(backend="zero"))
        self.uvi.start()

    def tearDown(self):
        self.uvi.stop()

    def test_zero_backend_returns_zeros(self):
        vx, vy = self.uvi.get_cmd()
        self.assertEqual(vx, 0.0)
        self.assertEqual(vy, 0.0)

    def test_zero_backend_normalized(self):
        vx_n, vy_n = self.uvi.get_cmd_normalized()
        self.assertEqual(vx_n, 0.0)
        self.assertEqual(vy_n, 0.0)

    def test_zero_backend_tensor(self):
        t = self.uvi.get_cmd_tensor(num_envs=4, device="cpu")
        self.assertEqual(t.shape, (4, 2))
        self.assertTrue(torch.all(t == 0.0))

    def test_zero_backend_no_thread(self):
        """Zero backend should not spawn a polling thread."""
        self.assertIsNone(self.uvi._thread)

    def test_stop_is_safe(self):
        """Calling stop on zero backend doesn't raise."""
        self.uvi.stop()
        self.uvi.stop()  # double stop also safe


class TestUserVelocityInputConfig(unittest.TestCase):
    """Config dataclass tests."""

    def test_default_config(self):
        cfg = UserVelocityInputCfg()
        self.assertEqual(cfg.backend, "pygame")
        self.assertEqual(cfg.max_vx, 0.30)
        self.assertEqual(cfg.max_vy, 0.30)
        self.assertEqual(cfg.deadband, 0.05)

    def test_custom_limits(self):
        cfg = UserVelocityInputCfg(max_vx=0.15, max_vy=0.20)
        self.assertEqual(cfg.max_vx, 0.15)
        self.assertEqual(cfg.max_vy, 0.20)

    def test_normalization_constants(self):
        """Scale factors must match action_term._CMD_SCALES[6:8]."""
        self.assertEqual(_VX_SCALE, 0.5)
        self.assertEqual(_VY_SCALE, 0.5)


class TestUserVelocityInputManual(unittest.TestCase):
    """Test internal state manipulation (simulating hardware input)."""

    def test_manual_set_and_read(self):
        uvi = UserVelocityInput(UserVelocityInputCfg(backend="zero"))
        uvi.start()
        # Simulate hardware input by writing internal state
        uvi._vx = 0.20
        uvi._vy = -0.10
        vx, vy = uvi.get_cmd()
        self.assertAlmostEqual(vx, 0.20)
        self.assertAlmostEqual(vy, -0.10)
        uvi.stop()

    def test_manual_normalized(self):
        uvi = UserVelocityInput(UserVelocityInputCfg(backend="zero"))
        uvi.start()
        uvi._vx = 0.25
        uvi._vy = -0.25
        vx_n, vy_n = uvi.get_cmd_normalized()
        self.assertAlmostEqual(vx_n, 0.50)   # 0.25 / 0.5
        self.assertAlmostEqual(vy_n, -0.50)
        uvi.stop()

    def test_tensor_broadcasts(self):
        uvi = UserVelocityInput(UserVelocityInputCfg(backend="zero"))
        uvi.start()
        uvi._vx = 0.10
        uvi._vy = 0.05
        t = uvi.get_cmd_tensor(num_envs=8, device="cpu")
        self.assertEqual(t.shape, (8, 2))
        # All envs should get same value
        self.assertAlmostEqual(t[0, 0].item(), 0.10 / 0.5, places=5)
        self.assertAlmostEqual(t[7, 1].item(), 0.05 / 0.5, places=5)
        uvi.stop()


# -----------------------------------------------------------------------
# CommandMixer tests
# -----------------------------------------------------------------------

class TestCommandMixerPassthrough(unittest.TestCase):
    def test_passthrough_returns_original(self):
        mixer = CommandMixer(CommandMixerCfg(mode="passthrough"))
        pi1 = torch.randn(4, 8)
        vel = torch.randn(4, 2)
        out = mixer.mix(pi1, vel)
        self.assertTrue(torch.equal(out, pi1))


class TestCommandMixerOverride(unittest.TestCase):
    def setUp(self):
        self.mixer = CommandMixer(CommandMixerCfg(mode="override"))
        self.N = 4

    def test_override_replaces_vx_vy(self):
        pi1 = torch.zeros(self.N, 8)
        pi1[:, 6] = 0.8  # pi1 wants vx=0.8
        pi1[:, 7] = -0.3
        vel = torch.tensor([[0.2, -0.5]]).expand(self.N, -1)
        out = self.mixer.mix(pi1, vel)
        # vx/vy should be user values
        self.assertTrue(torch.allclose(out[:, 6], torch.tensor(0.2)))
        self.assertTrue(torch.allclose(out[:, 7], torch.tensor(-0.5)))

    def test_override_preserves_other_dims(self):
        pi1 = torch.randn(self.N, 8)
        vel = torch.randn(self.N, 2)
        out = self.mixer.mix(pi1, vel)
        # Dims 0-5 unchanged
        self.assertTrue(torch.equal(out[:, :6], pi1[:, :6]))

    def test_override_does_not_modify_input(self):
        pi1 = torch.randn(self.N, 8)
        pi1_orig = pi1.clone()
        vel = torch.randn(self.N, 2)
        self.mixer.mix(pi1, vel)
        self.assertTrue(torch.equal(pi1, pi1_orig))


class TestCommandMixerBlend(unittest.TestCase):
    def test_blend_alpha_zero_equals_override(self):
        mixer = CommandMixer(CommandMixerCfg(mode="blend", blend_alpha=0.0))
        pi1 = torch.randn(2, 8)
        vel = torch.randn(2, 2)
        out = mixer.mix(pi1, vel)
        self.assertTrue(torch.allclose(out[:, 6], vel[:, 0]))
        self.assertTrue(torch.allclose(out[:, 7], vel[:, 1]))

    def test_blend_alpha_one_equals_pi1(self):
        mixer = CommandMixer(CommandMixerCfg(mode="blend", blend_alpha=1.0))
        pi1 = torch.randn(2, 8)
        vel = torch.randn(2, 2)
        out = mixer.mix(pi1, vel)
        self.assertTrue(torch.allclose(out[:, 6], pi1[:, 6]))
        self.assertTrue(torch.allclose(out[:, 7], pi1[:, 7]))

    def test_blend_alpha_half(self):
        mixer = CommandMixer(CommandMixerCfg(mode="blend", blend_alpha=0.5))
        pi1 = torch.ones(2, 8) * 0.4
        vel = torch.ones(2, 2) * 0.8
        out = mixer.mix(pi1, vel)
        expected = 0.5 * 0.4 + 0.5 * 0.8  # = 0.6
        self.assertAlmostEqual(out[0, 6].item(), expected, places=5)
        self.assertAlmostEqual(out[0, 7].item(), expected, places=5)

    def test_blend_preserves_other_dims(self):
        mixer = CommandMixer(CommandMixerCfg(mode="blend", blend_alpha=0.3))
        pi1 = torch.randn(3, 8)
        vel = torch.randn(3, 2)
        out = mixer.mix(pi1, vel)
        self.assertTrue(torch.equal(out[:, :6], pi1[:, :6]))


class TestCommandMixerConfig(unittest.TestCase):
    def test_default_config(self):
        cfg = CommandMixerCfg()
        self.assertEqual(cfg.mode, "override")
        self.assertEqual(cfg.blend_alpha, 0.0)
        self.assertEqual(cfg.vx_idx, 6)
        self.assertEqual(cfg.vy_idx, 7)

    def test_custom_indices(self):
        """If command layout ever changes, indices are configurable."""
        mixer = CommandMixer(CommandMixerCfg(mode="override", vx_idx=2, vy_idx=3))
        pi1 = torch.zeros(1, 8)
        vel = torch.tensor([[0.5, -0.5]])
        out = mixer.mix(pi1, vel)
        self.assertAlmostEqual(out[0, 2].item(), 0.5)
        self.assertAlmostEqual(out[0, 3].item(), -0.5)
        # Original indices 6,7 untouched
        self.assertEqual(out[0, 6].item(), 0.0)
        self.assertEqual(out[0, 7].item(), 0.0)


# -----------------------------------------------------------------------
# Teleop integration flow tests
# -----------------------------------------------------------------------

class TestTeleopFlow(unittest.TestCase):
    """End-to-end tests simulating the play_teleop.py flow:
    policy output → UserVelocityInput → CommandMixer → mixed actions."""

    def setUp(self):
        self.uvi = UserVelocityInput(UserVelocityInputCfg(backend="zero"))
        self.uvi.start()
        self.mixer = CommandMixer(CommandMixerCfg(mode="override"))
        self.num_envs = 4

    def tearDown(self):
        self.uvi.stop()

    def test_zero_input_preserves_pi1_vxy(self):
        """With zero backend and override mode, vx/vy should be 0 (user input)."""
        pi1_actions = torch.randn(self.num_envs, 8)
        vel_user = self.uvi.get_cmd_tensor(self.num_envs, "cpu")
        mixed = self.mixer.mix(pi1_actions, vel_user)
        # vx/vy zeroed by user
        self.assertTrue(torch.allclose(mixed[:, 6], torch.zeros(self.num_envs)))
        self.assertTrue(torch.allclose(mixed[:, 7], torch.zeros(self.num_envs)))
        # height/tilt from pi1 preserved
        self.assertTrue(torch.equal(mixed[:, :6], pi1_actions[:, :6]))

    def test_simulated_user_walk_forward(self):
        """Simulate user pushing joystick forward: vx=0.25 m/s, vy=0."""
        self.uvi._vx = 0.25
        self.uvi._vy = 0.0
        pi1_actions = torch.zeros(self.num_envs, 8)
        vel_user = self.uvi.get_cmd_tensor(self.num_envs, "cpu")
        mixed = self.mixer.mix(pi1_actions, vel_user)
        # 0.25 / 0.5 = 0.5 normalized
        self.assertAlmostEqual(mixed[0, 6].item(), 0.5, places=4)
        self.assertAlmostEqual(mixed[0, 7].item(), 0.0, places=4)

    def test_simulated_user_strafe_left(self):
        """Simulate user pushing joystick left: vx=0, vy=0.20 m/s."""
        self.uvi._vx = 0.0
        self.uvi._vy = 0.20
        pi1_actions = torch.randn(self.num_envs, 8)
        vel_user = self.uvi.get_cmd_tensor(self.num_envs, "cpu")
        mixed = self.mixer.mix(pi1_actions, vel_user)
        self.assertAlmostEqual(mixed[0, 6].item(), 0.0, places=4)
        self.assertAlmostEqual(mixed[0, 7].item(), 0.40, places=4)  # 0.20/0.5

    def test_blend_mode_teleop_flow(self):
        """Blend mode: pi1 wants vx=0.3 normalized, user wants vx=0.25 m/s.
        With alpha=0.5, result = 0.5*0.3 + 0.5*0.5 = 0.40."""
        mixer = CommandMixer(CommandMixerCfg(mode="blend", blend_alpha=0.5))
        self.uvi._vx = 0.25  # → 0.5 normalized
        pi1_actions = torch.zeros(self.num_envs, 8)
        pi1_actions[:, 6] = 0.3  # pi1 wants vx=0.3 normalized
        vel_user = self.uvi.get_cmd_tensor(self.num_envs, "cpu")
        mixed = mixer.mix(pi1_actions, vel_user)
        expected_vx = 0.5 * 0.3 + 0.5 * 0.5
        self.assertAlmostEqual(mixed[0, 6].item(), expected_vx, places=4)

    def test_passthrough_teleop_flow(self):
        """Passthrough mode: user input ignored, pi1 controls everything."""
        mixer = CommandMixer(CommandMixerCfg(mode="passthrough"))
        self.uvi._vx = 0.30
        self.uvi._vy = -0.30
        pi1_actions = torch.randn(self.num_envs, 8)
        vel_user = self.uvi.get_cmd_tensor(self.num_envs, "cpu")
        mixed = mixer.mix(pi1_actions, vel_user)
        self.assertTrue(torch.equal(mixed, pi1_actions))

    def test_max_speed_clamp(self):
        """User velocity capped at ±0.30 m/s → ±0.60 normalized."""
        self.uvi._vx = 0.30   # max
        self.uvi._vy = -0.30  # max negative
        vel_user = self.uvi.get_cmd_tensor(1, "cpu")
        self.assertAlmostEqual(vel_user[0, 0].item(), 0.60, places=4)
        self.assertAlmostEqual(vel_user[0, 1].item(), -0.60, places=4)

    def test_multi_step_consistency(self):
        """Mixer should produce consistent results across multiple calls
        (no internal state mutation)."""
        pi1_actions = torch.randn(self.num_envs, 8)
        self.uvi._vx = 0.15
        self.uvi._vy = 0.10
        vel = self.uvi.get_cmd_tensor(self.num_envs, "cpu")
        out1 = self.mixer.mix(pi1_actions, vel)
        out2 = self.mixer.mix(pi1_actions, vel)
        self.assertTrue(torch.equal(out1, out2))


if __name__ == "__main__":
    unittest.main()
