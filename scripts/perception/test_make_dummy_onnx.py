"""Tests for dummy ONNX model builder.

Verifies that make_dummy_onnx.py produces a valid ONNX model with the
correct input/output shapes and embedded detection values.

Run: pytest scripts/perception/test_make_dummy_onnx.py -x -q
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import onnx
from onnx import numpy_helper

# Import the builder
import importlib.util
import sys

_SCRIPT = os.path.join(
    os.path.dirname(__file__),
    "make_dummy_onnx.py",
)
_spec = importlib.util.spec_from_file_location("make_dummy_onnx", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["make_dummy_onnx"] = _mod
_spec.loader.exec_module(_mod)

make_dummy_yolo_onnx = _mod.make_dummy_yolo_onnx
_N_ANCHORS = _mod._N_ANCHORS
_INPUT_SIZE = _mod._INPUT_SIZE


class TestDummyOnnxModel(unittest.TestCase):
    """Test dummy ONNX model creation and structure."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._model_path = os.path.join(self._tmpdir, "test_model.onnx")
        make_dummy_yolo_onnx(self._model_path)

    def test_file_created(self):
        self.assertTrue(os.path.isfile(self._model_path))

    def test_valid_onnx(self):
        """Model passes ONNX checker."""
        model = onnx.load(self._model_path)
        onnx.checker.check_model(model)

    def test_input_shape(self):
        model = onnx.load(self._model_path)
        inp = model.graph.input[0]
        self.assertEqual(inp.name, "images")
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        self.assertEqual(dims, [1, 3, _INPUT_SIZE, _INPUT_SIZE])

    def test_output_shape(self):
        model = onnx.load(self._model_path)
        out = model.graph.output[0]
        self.assertEqual(out.name, "output0")
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        self.assertEqual(dims, [1, 5, _N_ANCHORS])

    def test_embedded_detection_values(self):
        """The constant output contains the configured detection."""
        model = onnx.load(self._model_path)
        # Find the Constant node
        const_node = None
        for node in model.graph.node:
            if node.op_type == "Constant":
                const_node = node
                break
        self.assertIsNotNone(const_node, "No Constant node found")

        tensor = numpy_helper.to_array(const_node.attribute[0].t)
        self.assertEqual(tensor.shape, (1, 5, _N_ANCHORS))

        # First detection slot: cx=320, cy=320, w=40, h=40, conf=0.95
        self.assertAlmostEqual(float(tensor[0, 0, 0]), 320.0, places=1)
        self.assertAlmostEqual(float(tensor[0, 1, 0]), 320.0, places=1)
        self.assertAlmostEqual(float(tensor[0, 2, 0]), 40.0, places=1)
        self.assertAlmostEqual(float(tensor[0, 3, 0]), 40.0, places=1)
        self.assertAlmostEqual(float(tensor[0, 4, 0]), 0.95, places=2)

    def test_other_slots_zero_conf(self):
        """All detection slots except the first have zero confidence."""
        model = onnx.load(self._model_path)
        for node in model.graph.node:
            if node.op_type == "Constant":
                tensor = numpy_helper.to_array(node.attribute[0].t)
                self.assertTrue(np.all(tensor[0, 4, 1:] == 0.0))
                break

    def test_custom_values(self):
        """Custom centre/confidence values are embedded correctly."""
        path = os.path.join(self._tmpdir, "custom.onnx")
        make_dummy_yolo_onnx(path, centre_x=100.0, centre_y=200.0, confidence=0.80)
        model = onnx.load(path)
        for node in model.graph.node:
            if node.op_type == "Constant":
                tensor = numpy_helper.to_array(node.attribute[0].t)
                self.assertAlmostEqual(float(tensor[0, 0, 0]), 100.0, places=1)
                self.assertAlmostEqual(float(tensor[0, 1, 0]), 200.0, places=1)
                self.assertAlmostEqual(float(tensor[0, 4, 0]), 0.80, places=2)
                break

    def test_subdirectory_creation(self):
        """Output directory is created if it doesn't exist."""
        path = os.path.join(self._tmpdir, "sub", "dir", "model.onnx")
        make_dummy_yolo_onnx(path)
        self.assertTrue(os.path.isfile(path))

    def test_returns_absolute_path(self):
        result = make_dummy_yolo_onnx(self._model_path)
        self.assertTrue(os.path.isabs(result))


class TestDummyOnnxWithOnnxruntime(unittest.TestCase):
    """Integration: load and run the dummy model with onnxruntime (if available)."""

    @classmethod
    def setUpClass(cls):
        try:
            import onnxruntime  # noqa: F401
            cls.ort_available = True
        except ImportError:
            cls.ort_available = False

    def setUp(self):
        if not self.ort_available:
            self.skipTest("onnxruntime not installed")
        self._tmpdir = tempfile.mkdtemp()
        self._model_path = os.path.join(self._tmpdir, "model.onnx")
        make_dummy_yolo_onnx(self._model_path)

    def test_load_and_run(self):
        """Model loads in onnxruntime and produces correct output shape."""
        import onnxruntime as ort

        session = ort.InferenceSession(self._model_path, providers=["CPUExecutionProvider"])
        inp_name = session.get_inputs()[0].name
        self.assertEqual(inp_name, "images")

        dummy_input = np.random.randn(1, 3, _INPUT_SIZE, _INPUT_SIZE).astype(np.float32)
        outputs = session.run(None, {inp_name: dummy_input})

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].shape, (1, 5, _N_ANCHORS))
        # Confidence of first slot should be 0.95 regardless of input
        self.assertAlmostEqual(float(outputs[0][0, 4, 0]), 0.95, places=2)

    def test_end_to_end_with_detector(self):
        """Full pipeline: load dummy model → detect → get 3D position."""
        import onnxruntime  # noqa: F401

        # Import BallDetector
        _PERC_DIR = os.path.normpath(os.path.join(
            os.path.dirname(__file__),
            "..", "..", "source", "go1_ball_balance", "go1_ball_balance", "perception",
        ))
        _cam_mod = importlib.util.spec_from_file_location(
            "perception.real.camera",
            os.path.join(_PERC_DIR, "real", "camera.py"),
        )
        cam_mod = importlib.util.module_from_spec(_cam_mod)
        sys.modules["perception.real.camera"] = cam_mod
        _cam_mod.loader.exec_module(cam_mod)

        _det_spec = importlib.util.spec_from_file_location(
            "perception.real.detector",
            os.path.join(_PERC_DIR, "real", "detector.py"),
        )
        det_mod = importlib.util.module_from_spec(_det_spec)
        sys.modules["perception.real.detector"] = det_mod
        _det_spec.loader.exec_module(det_mod)

        intrinsics = cam_mod.CameraIntrinsics(
            fx=425.19, fy=425.19, cx=423.36, cy=239.95,
            width=848, height=480,
        )

        # Create depth frame with ball at centre, 0.5m
        frame = np.full((480, 848), 500, dtype=np.uint16)  # 0.5m everywhere

        detector = det_mod.BallDetector(
            model_path=self._model_path,
            hough_fallback=False,
        )

        result = detector.detect(frame, intrinsics)
        self.assertIsNotNone(result)
        self.assertEqual(result.method, "yolo")
        self.assertAlmostEqual(result.confidence, 0.95, places=1)
        self.assertAlmostEqual(result.pos_cam[2], 0.50, delta=0.10)


if __name__ == "__main__":
    unittest.main()
