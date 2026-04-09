"""Tests for plot_training_curves.py."""

import os
import tempfile

import numpy as np
import pytest

from plot_training_curves import parse_training_log, smooth


@pytest.fixture
def sample_log(tmp_path):
    """Create a minimal RSL-RL training log."""
    log = tmp_path / "train.log"
    lines = []
    for i in range(50):
        lines.append(f"                       Mean reward: {50 + i * 0.5:.2f}")
        lines.append(f"               Mean episode length: {200 + i * 10:.2f}")
        lines.append(f"   Episode_Reward/ball_apex_height: {1.0 + i * 0.05:.4f}")
        lines.append(f"      Episode_Termination/time_out: {0.3 + i * 0.005:.4f}")
        lines.append(f"             Mean action noise std: {0.5 - i * 0.005:.2f}")
        lines.append(f"       Episode_Reward/ball_xy_dist: {-0.3 + i * 0.001:.4f}")
    log.write_text("\n".join(lines))
    return str(log)


def test_parse_log_counts(sample_log):
    m = parse_training_log(sample_log)
    assert len(m["timeout"]) == 50
    assert len(m["ep_length"]) == 50
    assert len(m["reward"]) == 50
    assert len(m["apex_reward"]) == 50
    assert len(m["noise_std"]) == 50


def test_parse_log_values(sample_log):
    m = parse_training_log(sample_log)
    assert abs(m["timeout"][0] - 0.3) < 1e-4
    assert abs(m["timeout"][-1] - (0.3 + 49 * 0.005)) < 1e-3
    assert abs(m["reward"][0] - 50.0) < 1e-2
    assert abs(m["ep_length"][0] - 200.0) < 1e-2


def test_parse_empty_log(tmp_path):
    log = tmp_path / "empty.log"
    log.write_text("some random text\nno metrics here\n")
    m = parse_training_log(str(log))
    assert len(m["timeout"]) == 0
    assert len(m["reward"]) == 0


def test_smooth_basic():
    arr = np.ones(100)
    s = smooth(arr, window=10)
    np.testing.assert_allclose(s, 1.0, atol=1e-10)


def test_smooth_short():
    arr = np.array([1.0, 2.0, 3.0])
    s = smooth(arr, window=10)
    # Should return unchanged if shorter than window
    assert len(s) == len(arr)


def test_plot_output(sample_log, tmp_path):
    from plot_training_curves import plot_training_curves

    out = str(tmp_path / "test_plot.png")
    plot_training_curves([sample_log], ["Test"], out)
    assert os.path.isfile(out)
    assert os.path.getsize(out) > 1000  # Non-trivial PNG


def test_plot_multi_log(sample_log, tmp_path):
    from plot_training_curves import plot_training_curves

    out = str(tmp_path / "test_multi.png")
    plot_training_curves([sample_log, sample_log], ["A", "B"], out)
    assert os.path.isfile(out)
