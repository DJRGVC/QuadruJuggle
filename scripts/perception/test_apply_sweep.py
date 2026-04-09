"""Tests for apply_sweep_results.py JSON parsing."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent / "apply_sweep_results.py"


def _make_sweep_json(tmp_path: Path, sweep_data: list, best_q: float) -> Path:
    """Create a sweep results JSON file for testing."""
    best_r = next(r for r in sweep_data if r["q_vel"] == best_q)
    data = {
        "sweep": sweep_data,
        "best_q_vel": best_q,
        "best_result": best_r,
        "config": {
            "q_vel_contact": 50.0,
            "q_vel_post_contact": 20.0,
            "post_contact_steps": 10,
            "target_height": 0.10,
            "num_envs": 512,
            "steps": 600,
            "warmup_steps": 50,
            "bisect_steps": 3,
        },
    }
    path = tmp_path / "sweep_q_vel_test.json"
    path.write_text(json.dumps(data, indent=2))
    return path


def _run_apply(json_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(json_path)],
        capture_output=True, text=True, timeout=10,
    )


@pytest.fixture
def valid_sweep(tmp_path):
    """Realistic sweep results with non-zero NIS/RMSE."""
    sweep = [
        {"q_vel": 0.4, "q_vel_contact": 50.0, "mean_nis": 52.9, "flight_nis": 65.3,
         "contact_nis": 1.2, "ekf_rmse_mm": 18.5, "raw_rmse_mm": 22.1,
         "improvement_pct": 16.3, "episodes": 445, "timeout_pct": 68.0},
        {"q_vel": 2.0, "q_vel_contact": 50.0, "mean_nis": 8.1, "flight_nis": 9.5,
         "contact_nis": 1.1, "ekf_rmse_mm": 14.2, "raw_rmse_mm": 22.1,
         "improvement_pct": 35.7, "episodes": 502, "timeout_pct": 68.0},
        {"q_vel": 5.0, "q_vel_contact": 50.0, "mean_nis": 3.2, "flight_nis": 3.8,
         "contact_nis": 1.0, "ekf_rmse_mm": 16.8, "raw_rmse_mm": 22.1,
         "improvement_pct": 24.0, "episodes": 498, "timeout_pct": 68.0},
        {"q_vel": 10.0, "q_vel_contact": 50.0, "mean_nis": 1.5, "flight_nis": 1.7,
         "contact_nis": 0.9, "ekf_rmse_mm": 20.5, "raw_rmse_mm": 22.1,
         "improvement_pct": 7.2, "episodes": 510, "timeout_pct": 68.0},
    ]
    return _make_sweep_json(tmp_path, sweep, best_q=5.0)


@pytest.fixture
def zeros_sweep(tmp_path):
    """All-zeros sweep (the pre-fix bug result)."""
    sweep = [
        {"q_vel": q, "q_vel_contact": 50.0, "mean_nis": 0, "flight_nis": 0,
         "contact_nis": 0, "ekf_rmse_mm": 0, "raw_rmse_mm": 0,
         "improvement_pct": 0, "episodes": 445, "timeout_pct": 0.0}
        for q in [0.4, 2.0, 5.0, 10.0]
    ]
    return _make_sweep_json(tmp_path, sweep, best_q=0.4)


class TestApplySweepResults:
    def test_valid_sweep_parses(self, valid_sweep):
        result = _run_apply(valid_sweep)
        assert result.returncode == 0
        # NIS=3.0 crossing between q_vel=5.0 (NIS=3.8) and q_vel=10.0 (NIS=1.7)
        assert "NIS=3.0 crossing" in result.stdout
        assert "EKF beats raw" in result.stdout

    def test_valid_sweep_shows_table(self, valid_sweep):
        result = _run_apply(valid_sweep)
        assert result.returncode == 0
        # Check table header
        assert "q_vel" in result.stdout
        assert "Flight" in result.stdout
        # Check that all q_vel values appear in table
        assert "0.4000" in result.stdout
        assert "5.0000" in result.stdout

    def test_valid_sweep_shows_recommendation(self, valid_sweep):
        result = _run_apply(valid_sweep)
        assert "Recommended BallEKFConfig changes:" in result.stdout
        # Recommended q_vel should be the NIS=3.0 crossing (interpolated)
        assert "q_vel =" in result.stdout
        assert "q_vel_contact = 50.0" in result.stdout

    def test_zeros_sweep_parses(self, zeros_sweep):
        """All-zeros sweep should still parse without crashing."""
        result = _run_apply(zeros_sweep)
        assert result.returncode == 0
        assert "Closest to NIS=3.0" in result.stdout

    def test_zeros_ekf_loses(self, zeros_sweep):
        """With 0 RMSE for both, EKF should not claim to beat raw."""
        result = _run_apply(zeros_sweep)
        # 0 == 0 → not strictly less → "LOSES to"
        assert "LOSES to" in result.stdout

    def test_missing_file_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "/nonexistent/file.json"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode != 0


class TestApplyToConfig:
    def test_patches_q_vel(self, tmp_path):
        """apply_to_config should update q_vel in a config file."""
        fake_config = tmp_path / "ball_ekf.py"
        fake_config.write_text(
            "    q_vel: float = 0.40     # velocity process noise std, FREE-FLIGHT (m/s) per sqrt(s)\n"
        )
        from apply_sweep_results import apply_to_config
        ok = apply_to_config(0.1234, config_path=str(fake_config))
        assert ok
        content = fake_config.read_text()
        assert "0.1234" in content
        assert "0.40" not in content
        assert "FREE-FLIGHT" in content

    def test_no_match_returns_false(self, tmp_path):
        """apply_to_config should return False if pattern not found."""
        fake_config = tmp_path / "ball_ekf.py"
        fake_config.write_text("# no q_vel here\n")
        from apply_sweep_results import apply_to_config
        ok = apply_to_config(0.1234, config_path=str(fake_config))
        assert not ok
