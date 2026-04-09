"""Create a minimal dummy ONNX model mimicking YOLOv8n ball-detector output.

The model is a trivial graph (Identity + Reshape) that produces the correct
output shape (1, 5, 8400) for a single-class YOLOv8 detector.  It does NOT
do real detection — it outputs a fixed synthetic detection at the image
centre with high confidence.  This is useful for:

  * Integration-testing BallDetector._load_model() + _detect_yolo() end-to-end
    (requires onnxruntime installed on the target).
  * Validating ONNX → TensorRT conversion on Jetson before real training.
  * Smoke-testing the depth→YOLO→3D pipeline without ultralytics.

Usage:
    python scripts/perception/make_dummy_onnx.py [--output models/dummy_yolo.onnx]

The output model:
  - Input:  "images" float32 (1, 3, 640, 640)
  - Output: "output0" float32 (1, 5, 8400)
  - The first detection slot contains a fixed bbox at (320, 320, 40, 40)
    with confidence 0.95.  All other slots have conf=0.

This matches the standard ultralytics YOLOv8 ONNX export format.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# YOLOv8 output: (1, 5, N_ANCHORS) for single class
# Rows: cx, cy, w, h, conf
_N_ANCHORS = 8400
_INPUT_SIZE = 640


def make_dummy_yolo_onnx(
    output_path: str,
    centre_x: float = 320.0,
    centre_y: float = 320.0,
    bbox_w: float = 40.0,
    bbox_h: float = 40.0,
    confidence: float = 0.95,
) -> str:
    """Build and save a dummy YOLOv8 ONNX model.

    Args:
        output_path: Where to write the .onnx file.
        centre_x, centre_y: Fixed detection centre in letterboxed pixels.
        bbox_w, bbox_h: Fixed bbox size in letterboxed pixels.
        confidence: Fixed confidence score.

    Returns:
        Absolute path to the written file.
    """
    # Build the fixed output tensor as a Constant node
    output_data = np.zeros((1, 5, _N_ANCHORS), dtype=np.float32)
    output_data[0, 0, 0] = centre_x  # cx
    output_data[0, 1, 0] = centre_y  # cy
    output_data[0, 2, 0] = bbox_w    # w
    output_data[0, 3, 0] = bbox_h    # h
    output_data[0, 4, 0] = confidence # conf

    # Graph: input → (ignored) → constant output
    # We use a Constant node for the output and Shape+Mul nodes to
    # create a dependency on the input (so ONNX doesn't optimise it away).
    input_tensor = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, [1, 3, _INPUT_SIZE, _INPUT_SIZE],
    )
    output_tensor = helper.make_tensor_value_info(
        "output0", TensorProto.FLOAT, [1, 5, _N_ANCHORS],
    )

    # Constant node for the fixed predictions
    const_tensor = numpy_helper.from_array(output_data, name="fixed_preds")
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["output0"],
        value=const_tensor,
    )

    # Shape node to consume the input (prevents ONNX from flagging unused input)
    shape_node = helper.make_node("Shape", inputs=["images"], outputs=["_input_shape"])

    graph = helper.make_graph(
        [shape_node, const_node],
        "dummy_yolo_ball",
        [input_tensor],
        [output_tensor],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    onnx.checker.check_model(model)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    onnx.save(model, output_path)

    return os.path.abspath(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create dummy YOLOv8 ONNX model for testing")
    parser.add_argument(
        "--output", "-o",
        default="models/dummy_yolo.onnx",
        help="Output path for .onnx file (default: models/dummy_yolo.onnx)",
    )
    parser.add_argument("--cx", type=float, default=320.0, help="Detection centre X")
    parser.add_argument("--cy", type=float, default=320.0, help="Detection centre Y")
    parser.add_argument("--conf", type=float, default=0.95, help="Detection confidence")
    args = parser.parse_args()

    path = make_dummy_yolo_onnx(args.output, centre_x=args.cx, centre_y=args.cy, confidence=args.conf)
    print(f"Saved dummy ONNX model to: {path}")
    print(f"  Input:  images float32 (1, 3, {_INPUT_SIZE}, {_INPUT_SIZE})")
    print(f"  Output: output0 float32 (1, 5, {_N_ANCHORS})")
    print(f"  Fixed detection: ({args.cx}, {args.cy}) conf={args.conf}")


if __name__ == "__main__":
    main()
