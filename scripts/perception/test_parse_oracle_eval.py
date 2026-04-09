"""Tests for parse_oracle_eval.py log parser."""
import textwrap
import pytest
from parse_oracle_eval import parse_log
import tempfile
import os


SAMPLE_LOG = textwrap.dedent("""\
    === Oracle pi1 eval: 1500 steps, 4 envs, target=0.42m ===
    [demo] Auto-detected pi2: /home/user/logs/model_best.pt
    [demo] Ball obs noise mode: oracle
    [demo] Fixed target apex height: 0.42m
    [demo] Camera world pos: [-0.08  0.    0.46]
    [demo] Camera world quat (ROS): [0.819  0.    -0.574  0.   ]
    [demo] Camera forward (world): [ 0.342  0.     0.94 ], elevation: 70.0°
    [demo] EKF initialized for 4 envs at world pos: [0.0 0.0 0.49]
    [demo] Applied 2.0 m/s upward velocity to ball.
    [demo] Running 1500 steps (mode=policy)...
    [demo] Step 0: det_rate=100%, ball_h=0.073m, det_rmse=0.0205m, ekf_rmse=0.0183m episodes=0 TO=0%
    [demo] Step 50: det_rate=72%, ball_h=0.320m, det_rmse=0.0189m, ekf_rmse=0.0210m episodes=1 TO=100%
    [demo] Step 100: det_rate=65%, ball_h=-0.020m, det_rmse=0.0195m, ekf_rmse=0.0250m episodes=2 TO=50%
    [demo] Step 150: det_rate=58%, ball_h=0.450m, det_rmse=0.0201m, ekf_rmse=0.0190m episodes=3 TO=66%

    [demo] SUMMARY (1500 steps, TRAINED POLICY):
      Detection rate: 870/1500 (58.0%)
      Detection RMSE: 0.0195 +/- 0.0082 m
      EKF RMSE: 0.0210 +/- 0.0095 m
      Episodes: 6  |  Timeout: 83.3%
    [demo] Trajectory data saved: /tmp/demo/trajectory.npz (1500 steps, 870 detections)
""")


@pytest.fixture
def log_file(tmp_path):
    p = tmp_path / "oracle_eval.log"
    p.write_text(SAMPLE_LOG)
    return str(p)


def test_parse_detection_rate(log_file):
    r = parse_log(log_file)
    assert r["det_count"] == 870
    assert r["total_steps"] == 1500
    assert abs(r["det_rate_pct"] - 58.0) < 0.1


def test_parse_rmse(log_file):
    r = parse_log(log_file)
    assert abs(r["det_rmse_mm"] - 19.5) < 0.1
    assert abs(r["ekf_rmse_mm"] - 21.0) < 0.1


def test_parse_episode_stats(log_file):
    r = parse_log(log_file)
    assert r["episodes"] == 6
    assert abs(r["timeout_pct"] - 83.3) < 0.1


def test_parse_camera_elevation(log_file):
    r = parse_log(log_file)
    assert abs(r["cam_elevation_deg"] - 70.0) < 0.1


def test_parse_noise_mode(log_file):
    r = parse_log(log_file)
    assert r["noise_mode"] == "oracle"


def test_parse_target_height(log_file):
    r = parse_log(log_file)
    assert abs(r["target_height_m"] - 0.42) < 0.01


def test_parse_step_logs(log_file):
    r = parse_log(log_file)
    assert len(r["step_logs"]) == 4
    assert r["step_logs"][0]["step"] == 0
    assert r["step_logs"][1]["ball_h_m"] == 0.320
    assert r["step_logs"][2]["episodes"] == 2


def test_parse_trajectory_npz(log_file):
    r = parse_log(log_file)
    assert r["trajectory_npz"] == "/tmp/demo/trajectory.npz"


def test_empty_log(tmp_path):
    p = tmp_path / "empty.log"
    p.write_text("[gpu_lock] GPU busy; waiting for other agent to finish...\n")
    r = parse_log(str(p))
    assert r == {}
