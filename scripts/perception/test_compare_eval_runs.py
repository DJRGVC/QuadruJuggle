"""Tests for compare_eval_runs.py."""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from compare_eval_runs import compare_runs, plot_comparison, plot_step_timeseries


_SAMPLE_LOG_ORACLE = """\
Ball obs noise mode: oracle
Fixed target apex height: 0.42m
Camera elevation: 70.0°
[demo] Step 0: det_rate=0%, ball_h=0.021m, det_rmse=0.0000m, ekf_rmse=0.0000m episodes=0 TO=0%
[demo] Step 50: det_rate=42%, ball_h=0.350m, det_rmse=0.0180m, ekf_rmse=0.0120m episodes=1 TO=100%
[demo] Step 100: det_rate=48%, ball_h=0.410m, det_rmse=0.0200m, ekf_rmse=0.0150m episodes=2 TO=100%

[demo] SUMMARY (100 steps, TRAINED POLICY):
  Detection rate: 48/100 (48.0%)
  Detection RMSE: 0.0190 +/- 0.0050 m
  EKF RMSE: 0.0140 +/- 0.0040 m
  Episodes: 2  |  Timeout: 100.0%
"""

_SAMPLE_LOG_D435I = """\
Ball obs noise mode: d435i
Fixed target apex height: 0.42m
Camera elevation: 70.0°
[demo] Step 0: det_rate=0%, ball_h=0.015m, det_rmse=0.0000m, ekf_rmse=0.0000m episodes=0 TO=0%
[demo] Step 50: det_rate=1%, ball_h=0.010m, det_rmse=0.0150m, ekf_rmse=0.0280m episodes=1 TO=0%

[demo] SUMMARY (100 steps, TRAINED POLICY):
  Detection rate: 1/100 (1.0%)
  Detection RMSE: 0.0150 +/- 0.0030 m
  EKF RMSE: 0.0280 +/- 0.0100 m
  Episodes: 2  |  Timeout: 0.0%
"""


@pytest.fixture
def oracle_log(tmp_path):
    p = tmp_path / "oracle.log"
    p.write_text(_SAMPLE_LOG_ORACLE)
    return str(p)


@pytest.fixture
def d435i_log(tmp_path):
    p = tmp_path / "d435i.log"
    p.write_text(_SAMPLE_LOG_D435I)
    return str(p)


def test_compare_runs_parses_both(oracle_log, d435i_log):
    results = compare_runs([oracle_log, d435i_log], ["Oracle", "D435i"])
    assert len(results) == 2
    assert results[0]["label"] == "Oracle"
    assert results[1]["label"] == "D435i"
    assert results[0]["det_rate_pct"] == 48.0
    assert results[1]["det_rate_pct"] == 1.0


def test_compare_runs_noise_mode(oracle_log, d435i_log):
    results = compare_runs([oracle_log, d435i_log], ["Oracle", "D435i"])
    assert results[0]["noise_mode"] == "oracle"
    assert results[1]["noise_mode"] == "d435i"


def test_compare_runs_rmse(oracle_log, d435i_log):
    results = compare_runs([oracle_log, d435i_log], ["Oracle", "D435i"])
    assert abs(results[0]["det_rmse_mm"] - 19.0) < 0.1
    assert abs(results[1]["det_rmse_mm"] - 15.0) < 0.1


def test_compare_runs_step_logs(oracle_log, d435i_log):
    results = compare_runs([oracle_log, d435i_log], ["Oracle", "D435i"])
    assert "step_logs" in results[0]
    assert len(results[0]["step_logs"]) == 3  # steps 0, 50, 100


def test_plot_comparison_creates_file(oracle_log, d435i_log, tmp_path):
    results = compare_runs([oracle_log, d435i_log], ["Oracle", "D435i"])
    out = str(tmp_path / "comparison.png")
    plot_comparison(results, out)
    assert os.path.isfile(out)
    assert os.path.getsize(out) > 1000


def test_plot_timeseries_creates_file(oracle_log, d435i_log, tmp_path):
    results = compare_runs([oracle_log, d435i_log], ["Oracle", "D435i"])
    out = str(tmp_path / "timeseries.png")
    plot_step_timeseries(results, out)
    ts_path = str(tmp_path / "timeseries_timeseries.png")
    assert os.path.isfile(ts_path)


def test_empty_log(tmp_path):
    empty = tmp_path / "empty.log"
    empty.write_text("[gpu_lock] GPU busy; waiting for other agent to finish...")
    results = compare_runs([str(empty)], ["Empty"])
    assert results[0]["label"] == "Empty"
    assert "det_rate_pct" not in results[0]


def test_single_run(oracle_log, tmp_path):
    results = compare_runs([oracle_log], ["Oracle"])
    out = str(tmp_path / "single.png")
    plot_comparison(results, out)
    assert os.path.isfile(out)
