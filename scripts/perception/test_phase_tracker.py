"""Tests for BallPhaseTracker."""

import sys
import os
import torch
import pytest

# Direct import to avoid Isaac Lab __init__.py chain (needs pxr/sim)
_PERCEPTION = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    "source", "go1_ball_balance", "go1_ball_balance", "perception",
))
sys.path.insert(0, _PERCEPTION)

from phase_tracker import (
    BallPhase,
    BallPhaseTracker,
    BallPhaseTrackerConfig,
)


@pytest.fixture
def tracker():
    return BallPhaseTracker(num_envs=4, device="cpu")


class TestInitialState:
    def test_initial_phase_is_contact(self, tracker):
        assert (tracker.phase == BallPhase.CONTACT).all()

    def test_initial_in_flight_false(self, tracker):
        assert not tracker.in_flight.any()

    def test_initial_bounce_count_zero(self, tracker):
        assert (tracker.bounce_count == 0).all()

    def test_initial_flight_fraction_zero(self, tracker):
        assert (tracker.flight_fraction == 0).all()


class TestContactToAscending:
    def test_launch_detected(self, tracker):
        """Ball leaves paddle with upward velocity → ASCENDING."""
        pos = torch.tensor([[0, 0, 0.55], [0, 0, 0.55], [0, 0, 0.40], [0, 0, 0.55]])
        vel = torch.tensor([[0, 0, 2.0], [0, 0, 2.0], [0, 0, 0.1], [0, 0, 2.0]])
        tracker.update(pos, vel, contact_z_threshold=0.50)

        # Envs 0,1,3 should be ascending (above threshold + upward vel)
        # Env 2 is below contact zone with low vel → stays contact
        assert tracker.phase[0] == BallPhase.ASCENDING
        assert tracker.phase[1] == BallPhase.ASCENDING
        assert tracker.phase[2] == BallPhase.CONTACT
        assert tracker.phase[3] == BallPhase.ASCENDING

    def test_launch_increments_bounce_count(self, tracker):
        pos = torch.tensor([[0, 0, 0.55]] * 4)
        vel = torch.tensor([[0, 0, 2.0]] * 4)
        tracker.update(pos, vel, contact_z_threshold=0.50)
        assert (tracker.bounce_count == 1).all()

    def test_no_launch_below_vz_threshold(self, tracker):
        """Ball above contact zone but slow upward vel → stays contact."""
        pos = torch.tensor([[0, 0, 0.55]] * 4)
        vel = torch.tensor([[0, 0, 0.3]] * 4)  # below default 0.5 threshold
        tracker.update(pos, vel, contact_z_threshold=0.50)
        assert (tracker.phase == BallPhase.CONTACT).all()

    def test_no_launch_in_contact_zone(self, tracker):
        """Ball in contact zone with upward vel → stays contact."""
        pos = torch.tensor([[0, 0, 0.49]] * 4)
        vel = torch.tensor([[0, 0, 2.0]] * 4)
        tracker.update(pos, vel, contact_z_threshold=0.50)
        assert (tracker.phase == BallPhase.CONTACT).all()


class TestAscendingToDescending:
    def test_apex_transition(self, tracker):
        """Ball reaches apex (vz ≤ 0) → DESCENDING."""
        pos = torch.tensor([[0, 0, 0.55]] * 4)
        vel = torch.tensor([[0, 0, 2.0]] * 4)
        tracker.update(pos, vel, contact_z_threshold=0.50)
        assert (tracker.phase == BallPhase.ASCENDING).all()

        # At apex
        pos2 = torch.tensor([[0, 0, 0.70]] * 4)
        vel2 = torch.tensor([[0, 0, 0.0]] * 4)
        tracker.update(pos2, vel2, contact_z_threshold=0.50)
        assert (tracker.phase == BallPhase.DESCENDING).all()

    def test_peak_height_tracked(self, tracker):
        pos = torch.tensor([[0, 0, 0.55]] * 4)
        vel = torch.tensor([[0, 0, 2.0]] * 4)
        tracker.update(pos, vel, contact_z_threshold=0.50)

        pos2 = torch.tensor([[0, 0, 0.80]] * 4)
        vel2 = torch.tensor([[0, 0, 1.0]] * 4)
        tracker.update(pos2, vel2, contact_z_threshold=0.50)

        assert torch.allclose(tracker.peak_height, torch.tensor([0.80] * 4))


class TestDescendingToContact:
    def test_landing_transition(self, tracker):
        """Ball falls into contact zone → CONTACT."""
        # Launch
        tracker.update(
            torch.tensor([[0, 0, 0.55]] * 4),
            torch.tensor([[0, 0, 2.0]] * 4),
            contact_z_threshold=0.50,
        )
        # Apex
        tracker.update(
            torch.tensor([[0, 0, 0.80]] * 4),
            torch.tensor([[0, 0, 0.0]] * 4),
            contact_z_threshold=0.50,
        )
        assert (tracker.phase == BallPhase.DESCENDING).all()

        # Landing
        tracker.update(
            torch.tensor([[0, 0, 0.48]] * 4),
            torch.tensor([[0, 0, -2.0]] * 4),
            contact_z_threshold=0.50,
        )
        assert (tracker.phase == BallPhase.CONTACT).all()


class TestFullBounceSequence:
    def test_three_bounces(self, tracker):
        """Simulate 3 complete bounce cycles."""
        cz = 0.50
        for bounce in range(3):
            # Launch
            tracker.update(
                torch.tensor([[0, 0, cz + 0.05]] * 4),
                torch.tensor([[0, 0, 3.0]] * 4),
                contact_z_threshold=cz,
            )
            assert (tracker.phase == BallPhase.ASCENDING).all()

            # Ascending
            tracker.update(
                torch.tensor([[0, 0, cz + 0.30]] * 4),
                torch.tensor([[0, 0, 1.5]] * 4),
                contact_z_threshold=cz,
            )

            # Apex
            tracker.update(
                torch.tensor([[0, 0, cz + 0.50]] * 4),
                torch.tensor([[0, 0, -0.1]] * 4),
                contact_z_threshold=cz,
            )
            assert (tracker.phase == BallPhase.DESCENDING).all()

            # Landing
            tracker.update(
                torch.tensor([[0, 0, cz - 0.02]] * 4),
                torch.tensor([[0, 0, -3.0]] * 4),
                contact_z_threshold=cz,
            )
            assert (tracker.phase == BallPhase.CONTACT).all()

        assert (tracker.bounce_count == 3).all()

    def test_flight_fraction_after_bounces(self, tracker):
        cz = 0.50
        # 2 steps contact, then bounce, 3 steps flight, land
        tracker.update(  # step 1: contact
            torch.tensor([[0, 0, 0.49]] * 4),
            torch.tensor([[0, 0, 0.0]] * 4),
            contact_z_threshold=cz,
        )
        tracker.update(  # step 2: contact
            torch.tensor([[0, 0, 0.49]] * 4),
            torch.tensor([[0, 0, 0.0]] * 4),
            contact_z_threshold=cz,
        )
        tracker.update(  # step 3: launch
            torch.tensor([[0, 0, 0.55]] * 4),
            torch.tensor([[0, 0, 3.0]] * 4),
            contact_z_threshold=cz,
        )
        tracker.update(  # step 4: ascending
            torch.tensor([[0, 0, 0.70]] * 4),
            torch.tensor([[0, 0, 1.0]] * 4),
            contact_z_threshold=cz,
        )
        tracker.update(  # step 5: descending
            torch.tensor([[0, 0, 0.65]] * 4),
            torch.tensor([[0, 0, -1.0]] * 4),
            contact_z_threshold=cz,
        )
        # 2 contact + 3 flight = 5 steps total → flight fraction = 3/5 = 0.6
        expected = 3.0 / 5.0
        assert torch.allclose(tracker.flight_fraction, torch.tensor([expected] * 4), atol=0.01)


class TestMixedEnvs:
    def test_independent_per_env(self, tracker):
        """Different envs in different phases simultaneously."""
        pos = torch.tensor([
            [0, 0, 0.49],  # contact zone
            [0, 0, 0.55],  # above threshold, fast
            [0, 0, 0.80],  # high up
            [0, 0, 0.49],  # contact zone
        ])
        vel = torch.tensor([
            [0, 0, 0.0],   # stationary → contact
            [0, 0, 3.0],   # upward → ascending
            [0, 0, 0.0],   # stationary high → contact (hasn't launched)
            [0, 0, -2.0],  # downward → contact (in zone)
        ])
        tracker.update(pos, vel, contact_z_threshold=0.50)

        assert tracker.phase[0] == BallPhase.CONTACT
        assert tracker.phase[1] == BallPhase.ASCENDING
        assert tracker.phase[2] == BallPhase.CONTACT  # never launched
        assert tracker.phase[3] == BallPhase.CONTACT


class TestReset:
    def test_reset_clears_state(self, tracker):
        # Do a bounce
        tracker.update(
            torch.tensor([[0, 0, 0.55]] * 4),
            torch.tensor([[0, 0, 2.0]] * 4),
            contact_z_threshold=0.50,
        )
        assert (tracker.bounce_count == 1).all()

        # Reset envs 0 and 2
        tracker.reset(torch.tensor([0, 2]))
        assert tracker.bounce_count[0] == 0
        assert tracker.bounce_count[1] == 1
        assert tracker.bounce_count[2] == 0
        assert tracker.bounce_count[3] == 1
        assert tracker.phase[0] == BallPhase.CONTACT
        assert tracker.phase[1] == BallPhase.ASCENDING


class TestSummary:
    def test_summary_returns_dict(self, tracker):
        s = tracker.summary()
        assert "mean_bounces" in s
        assert "mean_flight_fraction" in s
        assert "mean_peak_height" in s
        assert "pct_in_flight" in s
        assert "pct_contact" in s

    def test_summary_values_consistent(self, tracker):
        tracker.update(
            torch.tensor([[0, 0, 0.55]] * 4),
            torch.tensor([[0, 0, 2.0]] * 4),
            contact_z_threshold=0.50,
        )
        s = tracker.summary()
        assert s["pct_in_flight"] == pytest.approx(100.0)
        assert s["pct_contact"] == pytest.approx(0.0)
        assert s["mean_bounces"] == pytest.approx(1.0)


class TestTensorThreshold:
    def test_per_env_threshold(self, tracker):
        """contact_z_threshold can be a per-env tensor."""
        pos = torch.tensor([
            [0, 0, 0.55],
            [0, 0, 0.55],
            [0, 0, 0.55],
            [0, 0, 0.55],
        ])
        vel = torch.tensor([[0, 0, 2.0]] * 4)
        # Different thresholds per env
        cz = torch.tensor([0.50, 0.60, 0.50, 0.60])
        tracker.update(pos, vel, contact_z_threshold=cz)

        # Envs 0,2: above 0.50+margin → ascending
        # Envs 1,3: below 0.60+margin → contact (0.55 < 0.61)
        assert tracker.phase[0] == BallPhase.ASCENDING
        assert tracker.phase[1] == BallPhase.CONTACT
        assert tracker.phase[2] == BallPhase.ASCENDING
        assert tracker.phase[3] == BallPhase.CONTACT


class TestWeakBounce:
    def test_weak_bounce_returns_to_contact(self, tracker):
        """Ball barely leaves contact zone then falls back."""
        cz = 0.50
        # Launch with just enough velocity
        tracker.update(
            torch.tensor([[0, 0, cz + 0.02]] * 4),
            torch.tensor([[0, 0, 0.6]] * 4),
            contact_z_threshold=cz,
        )
        assert (tracker.phase == BallPhase.ASCENDING).all()

        # Falls back into contact zone without reaching descending
        tracker.update(
            torch.tensor([[0, 0, cz - 0.01]] * 4),
            torch.tensor([[0, 0, -0.5]] * 4),
            contact_z_threshold=cz,
        )
        assert (tracker.phase == BallPhase.CONTACT).all()


class TestCustomConfig:
    def test_high_launch_threshold(self):
        """Higher launch_vz_threshold filters out weak bounces."""
        cfg = BallPhaseTrackerConfig(launch_vz_threshold=2.0)
        tracker = BallPhaseTracker(num_envs=2, cfg=cfg)

        pos = torch.tensor([[0, 0, 0.55], [0, 0, 0.55]])
        vel = torch.tensor([[0, 0, 1.5], [0, 0, 3.0]])
        tracker.update(pos, vel, contact_z_threshold=0.50)

        assert tracker.phase[0] == BallPhase.CONTACT  # 1.5 < 2.0 threshold
        assert tracker.phase[1] == BallPhase.ASCENDING  # 3.0 > 2.0

    def test_large_flight_margin(self):
        """Large flight margin requires ball to be higher before counting as flight."""
        cfg = BallPhaseTrackerConfig(flight_height_margin=0.05)
        tracker = BallPhaseTracker(num_envs=2, cfg=cfg)

        pos = torch.tensor([[0, 0, 0.53], [0, 0, 0.56]])
        vel = torch.tensor([[0, 0, 2.0], [0, 0, 2.0]])
        tracker.update(pos, vel, contact_z_threshold=0.50)

        # 0.53 < 0.50 + 0.05 = 0.55 → not above flight threshold
        assert tracker.phase[0] == BallPhase.CONTACT
        # 0.56 > 0.55 → above flight threshold
        assert tracker.phase[1] == BallPhase.ASCENDING
