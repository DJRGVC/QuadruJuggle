"""Tests for plot_training_tb.py."""

import os
import struct
import tempfile

import numpy as np
import pytest

from plot_training_tb import load_tb_scalars, smooth


class TestSmooth:
    def test_basic(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = smooth(arr, window=3)
        assert len(result) == 3
        np.testing.assert_allclose(result, [2.0, 3.0, 4.0])

    def test_short_array(self):
        arr = np.array([1.0, 2.0])
        result = smooth(arr, window=5)
        np.testing.assert_array_equal(result, arr)

    def test_window_1(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = smooth(arr, window=1)
        np.testing.assert_allclose(result, arr)


class TestLoadTbScalars:
    def test_missing_dir(self):
        result = load_tb_scalars("/nonexistent", ["some/tag"])
        assert result == {"some/tag": []}

    def test_missing_tag(self, tmp_path):
        # Empty dir — no events
        result = load_tb_scalars(str(tmp_path), ["nonexistent/tag"])
        assert result == {"nonexistent/tag": []}


class TestPlotTrainingTb:
    def test_import(self):
        from plot_training_tb import plot_training_tb
        assert callable(plot_training_tb)

    def test_main_import(self):
        from plot_training_tb import main
        assert callable(main)
