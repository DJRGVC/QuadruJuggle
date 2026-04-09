"""Fine-tune YOLOv8n+P2 on synthetic depth data and export to ONNX.

Wraps ultralytics YOLOv8 training API for the ball detection task.
Produces an ONNX model ready for BallDetector._detect_yolo().

Usage:
    # Generate synthetic data first:
    python scripts/perception/generate_yolo_data.py --out data/yolo_ball --n 5000

    # Train and export:
    python scripts/perception/train_yolo_ball.py \
        --data data/yolo_ball --epochs 100 --imgsz 640

    # The ONNX model is saved to data/yolo_ball/best.onnx

Requires: pip install ultralytics
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

try:
    import cv2  # type: ignore[import-untyped]
except ImportError:
    cv2 = None


def split_dataset(
    data_dir: str | Path,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Split images/labels into train/val sets.

    Moves files into data_dir/{train,val}/{images,labels}/ and returns
    the paths. Idempotent — skips if train/ already exists.

    Returns:
        (train_dir, val_dir) paths.
    """
    data = Path(data_dir)
    train_dir = data / "train"
    val_dir = data / "val"

    # Skip if already split
    if train_dir.exists() and val_dir.exists():
        return train_dir, val_dir

    img_dir = data / "images"
    lbl_dir = data / "labels"

    if not img_dir.exists():
        raise FileNotFoundError(f"No images/ directory in {data}")

    # Collect image stems
    stems = sorted(
        p.stem for p in img_dir.iterdir() if p.suffix in (".png", ".jpg")
    )
    if not stems:
        raise ValueError(f"No images found in {img_dir}")

    rng = np.random.default_rng(seed)
    rng.shuffle(stems)

    n_val = max(1, int(len(stems) * val_fraction))
    val_stems = set(stems[:n_val])
    train_stems = set(stems[n_val:])

    for split_name, split_stems in [("train", train_stems), ("val", val_stems)]:
        split = data / split_name
        (split / "images").mkdir(parents=True, exist_ok=True)
        (split / "labels").mkdir(parents=True, exist_ok=True)

        for stem in split_stems:
            # Move image
            for ext in (".png", ".jpg"):
                src = img_dir / f"{stem}{ext}"
                if src.exists():
                    shutil.move(str(src), str(split / "images" / src.name))
                    break

            # Move label
            lbl_src = lbl_dir / f"{stem}.txt"
            if lbl_src.exists():
                shutil.move(str(lbl_src), str(split / "labels" / lbl_src.name))

    # Clean up empty source dirs
    if img_dir.exists() and not any(img_dir.iterdir()):
        img_dir.rmdir()
    if lbl_dir.exists() and not any(lbl_dir.iterdir()):
        lbl_dir.rmdir()

    return train_dir, val_dir


def write_dataset_yaml(
    data_dir: str | Path,
    train_dir: str | Path,
    val_dir: str | Path,
) -> Path:
    """Write/overwrite dataset.yaml for ultralytics.

    Returns path to the written yaml file.
    """
    data = Path(data_dir)
    yaml_path = data / "dataset.yaml"

    content = (
        f"path: {data.resolve()}\n"
        f"train: {Path(train_dir).relative_to(data)}/images\n"
        f"val: {Path(val_dir).relative_to(data)}/images\n"
        f"nc: 1\n"
        f"names: ['ball']\n"
    )
    yaml_path.write_text(content)
    return yaml_path


def depth_png_to_rgb(src_dir: Path) -> None:
    """Convert 16-bit depth PNGs to 3-channel 8-bit for YOLO.

    YOLOv8 expects 8-bit RGB images. Our synthetic data is uint16 depth
    in millimetres. This converts in-place: depth -> normalised 8-bit,
    replicated to 3 channels.

    Conversion: map [168mm, 2000mm] -> [0, 255], invalid (0) stays 0.
    """
    if cv2 is None:
        raise ImportError("cv2 required for depth conversion")

    min_mm, max_mm = 168, 2000

    for png in sorted(src_dir.glob("*.png")):
        img = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
        if img is None or img.dtype != np.uint16:
            continue

        # Normalise to 8-bit
        depth_f = img.astype(np.float32)
        valid = (img >= min_mm) & (img <= max_mm)
        img8 = np.zeros(img.shape, dtype=np.uint8)
        img8[valid] = np.clip(
            (depth_f[valid] - min_mm) * 255.0 / (max_mm - min_mm),
            0, 255,
        ).astype(np.uint8)

        # Stack to 3-channel (YOLOv8 expects RGB)
        img_rgb = np.stack([img8, img8, img8], axis=-1)
        cv2.imwrite(str(png), img_rgb)


def train(
    data_dir: str,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    model_name: str = "yolov8n.pt",
    device: str | None = None,
    workers: int = 4,
    patience: int = 20,
) -> Path:
    """Train YOLOv8 on the synthetic ball dataset and export to ONNX.

    Args:
        data_dir: Path to dataset (with images/ and labels/ or already split).
        epochs: Training epochs.
        imgsz: Input image size.
        batch: Batch size.
        model_name: Base YOLOv8 model to fine-tune from.
        device: Device string (e.g. "0" for GPU, "cpu"). None = auto.
        workers: Dataloader workers.
        patience: Early stopping patience (epochs without mAP improvement).

    Returns:
        Path to exported ONNX model.
    """
    try:
        from ultralytics import YOLO  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "ultralytics is required for YOLO training. "
            "Install with: pip install ultralytics"
        )

    data = Path(data_dir)

    # Step 1: Split dataset if not already split
    train_dir, val_dir = split_dataset(data)

    # Step 2: Convert depth PNGs to 8-bit RGB (ultralytics expects 8-bit)
    print("Converting depth images to 8-bit RGB...")
    depth_png_to_rgb(train_dir / "images")
    depth_png_to_rgb(val_dir / "images")

    # Step 3: Write dataset.yaml pointing to split dirs
    yaml_path = write_dataset_yaml(data, train_dir, val_dir)
    print(f"Dataset YAML: {yaml_path}")

    # Step 4: Train
    model = YOLO(model_name)
    train_kwargs: dict = dict(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        patience=patience,
        project=str(data / "runs"),
        name="ball_detect",
        exist_ok=True,
        # Depth-specific augmentation: no colour jitter (meaningless for depth),
        # but geometric augs are useful.
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        flipud=0.5,    # ball can appear from any direction
        fliplr=0.5,
        mosaic=0.0,     # disable mosaic — single-object detection
        mixup=0.0,      # disable mixup — single class, single object
    )
    if device is not None:
        train_kwargs["device"] = device

    results = model.train(**train_kwargs)
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Best checkpoint: {best_pt}")

    # Step 5: Export best model to ONNX
    best_model = YOLO(str(best_pt))
    onnx_path_str = best_model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=True,
        opset=17,
    )
    onnx_path = Path(onnx_path_str)

    # Copy ONNX to a stable location
    final_onnx = data / "best.onnx"
    shutil.copy2(str(onnx_path), str(final_onnx))
    print(f"ONNX model exported: {final_onnx}")

    return final_onnx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 on synthetic ball depth data"
    )
    parser.add_argument("--data", type=str, default="data/yolo_ball",
                        help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Base model to fine-tune")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (e.g. 0, cpu)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    args = parser.parse_args()

    train(
        data_dir=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model_name=args.model,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
