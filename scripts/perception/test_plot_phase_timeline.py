"""Tests for plot_phase_timeline.py."""

import os
import tempfile

import numpy as np
import pytest

from plot_phase_timeline import (
    ASCENDING,
    CONTACT,
    DESCENDING,
    compute_bounce_events,
    compute_per_step_error,
    compute_phase_spans,
    load_npz,
    plot_phase_timeline,
)


def _make_traj(n=200, n_det=20, with_phase=True, with_sched=True):
    """Create a synthetic trajectory dict."""
    t_steps = np.arange(n)
    gt = np.zeros((n, 3))
    # Simulate a ball bouncing: parabolic arcs
    for i in range(n):
        cycle = i % 50
        gt[i, 2] = 0.47 + max(0, 0.3 * (cycle / 25 - (cycle / 25) ** 2))
    gt[:, 0] = np.random.normal(0, 0.01, n)
    gt[:, 1] = np.random.normal(0, 0.01, n)

    ekf = gt + np.random.normal(0, 0.005, (n, 3))

    # Phase: contact when height near paddle, ascending/descending otherwise
    phase = np.full(n, CONTACT, dtype=int)
    if with_phase:
        for i in range(n):
            h = gt[i, 2] - 0.47
            if h > 0.01:
                cycle = i % 50
                phase[i] = ASCENDING if cycle < 25 else DESCENDING

    # Detections at random steps
    det_idx = np.sort(np.random.choice(n, size=min(n_det, n), replace=False))
    det = gt[det_idx] + np.random.normal(0, 0.01, (len(det_idx), 3))

    traj = {
        "gt": gt,
        "ekf": ekf,
        "steps": t_steps,
        "dt": 0.005,
        "det": det,
        "det_steps": det_idx.astype(float),
        "ball_h": gt[:, 2] - 0.47,
        "anchored_step": (phase == CONTACT).astype(int),
    }
    if with_phase:
        traj["phase"] = phase
    if with_sched:
        traj["sched_active"] = (phase != CONTACT).astype(int) if with_phase else np.ones(n, dtype=int)
    return traj


class TestComputePhaseSpans:
    def test_empty(self):
        assert compute_phase_spans(np.array([])) == []

    def test_single_phase(self):
        phase = np.array([0, 0, 0, 0])
        spans = compute_phase_spans(phase)
        assert spans == [(0, 4, 0)]

    def test_alternating(self):
        phase = np.array([0, 0, 1, 1, 2, 2, 0, 0])
        spans = compute_phase_spans(phase)
        assert spans == [(0, 2, 0), (2, 4, 1), (4, 6, 2), (6, 8, 0)]

    def test_single_step(self):
        phase = np.array([2])
        assert compute_phase_spans(phase) == [(0, 1, 2)]


class TestComputeBounceEvents:
    def test_empty(self):
        result = compute_bounce_events(np.array([]))
        assert len(result) == 0

    def test_no_bounces(self):
        phase = np.array([0, 0, 0, 0])
        assert len(compute_bounce_events(phase)) == 0

    def test_single_bounce(self):
        phase = np.array([0, 0, 1, 1, 2, 2, 0, 0])
        bounces = compute_bounce_events(phase)
        assert list(bounces) == [2]

    def test_multiple_bounces(self):
        phase = np.array([0, 1, 2, 0, 1, 2, 0])
        bounces = compute_bounce_events(phase)
        assert list(bounces) == [1, 4]

    def test_descending_to_ascending(self):
        phase = np.array([2, 2, 1, 1])
        bounces = compute_bounce_events(phase)
        assert list(bounces) == [2]


class TestComputePerStepError:
    def test_zero_error(self):
        gt = np.ones((5, 3))
        ekf = np.ones((5, 3))
        err = compute_per_step_error(gt, ekf)
        np.testing.assert_allclose(err, 0.0)

    def test_known_error(self):
        gt = np.array([[0, 0, 0]])
        ekf = np.array([[0.001, 0, 0]])
        err = compute_per_step_error(gt, ekf)
        np.testing.assert_allclose(err, 1.0)  # 1mm


class TestLoadNpz:
    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_npz("/nonexistent/path.npz")

    def test_loads_keys(self, tmp_path):
        path = str(tmp_path / "test.npz")
        np.savez_compressed(path, gt=np.zeros((10, 3)), ekf=np.ones((10, 3)))
        data = load_npz(path)
        assert "gt" in data
        assert "ekf" in data
        assert data["gt"].shape == (10, 3)


class TestPlotPhaseTimeline:
    def test_basic_plot(self, tmp_path):
        traj = _make_traj(n=200)
        out = str(tmp_path / "test_timeline.png")
        plot_phase_timeline(traj, out)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 1000

    def test_no_phase_data(self, tmp_path):
        traj = _make_traj(n=100, with_phase=False)
        out = str(tmp_path / "no_phase.png")
        plot_phase_timeline(traj, out)
        assert os.path.isfile(out)

    def test_no_scheduling_data(self, tmp_path):
        traj = _make_traj(n=100, with_sched=False)
        out = str(tmp_path / "no_sched.png")
        plot_phase_timeline(traj, out)
        assert os.path.isfile(out)

    def test_no_detections(self, tmp_path):
        traj = _make_traj(n=100, n_det=0)
        traj["det"] = np.zeros((0, 3))
        traj["det_steps"] = np.array([])
        out = str(tmp_path / "no_det.png")
        plot_phase_timeline(traj, out)
        assert os.path.isfile(out)

    def test_no_anchors(self, tmp_path):
        traj = _make_traj(n=100)
        traj["anchored_step"] = np.zeros(100, dtype=int)
        out = str(tmp_path / "no_anchor.png")
        plot_phase_timeline(traj, out)
        assert os.path.isfile(out)

    def test_short_trajectory(self, tmp_path):
        traj = _make_traj(n=10, n_det=2)
        out = str(tmp_path / "short.png")
        plot_phase_timeline(traj, out)
        assert os.path.isfile(out)

    def test_custom_title(self, tmp_path):
        traj = _make_traj(n=50)
        out = str(tmp_path / "titled.png")
        plot_phase_timeline(traj, out, title="Test Title")
        assert os.path.isfile(out)

    def test_creates_output_dir(self, tmp_path):
        traj = _make_traj(n=50)
        out = str(tmp_path / "subdir" / "nested" / "plot.png")
        plot_phase_timeline(traj, out)
        assert os.path.isfile(out)
