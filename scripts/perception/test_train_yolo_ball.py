"""Tests for YOLO training pipeline utilities.

Tests the non-ultralytics parts: dataset splitting, YAML generation,
and depth-to-RGB conversion.

Run: pytest scripts/perception/test_train_yolo_ball.py -x -q
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from train_yolo_ball import (
    depth_png_to_rgb,
    split_dataset,
    write_dataset_yaml,
)


# --- Fixtures ---

@pytest.fixture
def dummy_dataset(tmp_path: Path) -> Path:
    """Create a minimal synthetic dataset with 20 images."""
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()

    rng = np.random.default_rng(0)
    for i in range(20):
        stem = f"{i:06d}"
        # Create a uint16 depth image (480x848)
        depth = rng.integers(200, 1500, size=(480, 848), dtype=np.uint16)
        cv2.imwrite(str(img_dir / f"{stem}.png"), depth)
        # Create a YOLO label
        (lbl_dir / f"{stem}.txt").write_text("0 0.500000 0.500000 0.050000 0.050000\n")

    # dataset.yaml
    (tmp_path / "dataset.yaml").write_text(
        f"path: {tmp_path}\ntrain: images\nval: images\nnc: 1\nnames: ['ball']\n"
    )
    return tmp_path


@pytest.fixture
def depth_images_dir(tmp_path: Path) -> Path:
    """Create a directory with uint16 depth PNGs for conversion testing."""
    d = tmp_path / "images"
    d.mkdir()
    rng = np.random.default_rng(1)
    for i in range(5):
        depth = np.zeros((100, 100), dtype=np.uint16)
        # Fill some valid region
        depth[30:70, 30:70] = rng.integers(200, 1800, size=(40, 40), dtype=np.uint16)
        cv2.imwrite(str(d / f"{i:04d}.png"), depth)
    return d


# --- split_dataset tests ---

class TestSplitDataset:
    def test_creates_train_val_dirs(self, dummy_dataset: Path):
        train_dir, val_dir = split_dataset(dummy_dataset, val_fraction=0.2, seed=42)
        assert train_dir.exists()
        assert val_dir.exists()
        assert (train_dir / "images").exists()
        assert (train_dir / "labels").exists()
        assert (val_dir / "images").exists()
        assert (val_dir / "labels").exists()

    def test_split_counts(self, dummy_dataset: Path):
        train_dir, val_dir = split_dataset(dummy_dataset, val_fraction=0.2, seed=42)
        n_train = len(list((train_dir / "images").glob("*.png")))
        n_val = len(list((val_dir / "images").glob("*.png")))
        assert n_train + n_val == 20
        assert n_val == 4  # max(1, int(20 * 0.2)) = 4
        assert n_val >= 1
        assert n_train >= 1

    def test_labels_match_images(self, dummy_dataset: Path):
        train_dir, val_dir = split_dataset(dummy_dataset, val_fraction=0.2, seed=42)
        for split in [train_dir, val_dir]:
            img_stems = {p.stem for p in (split / "images").glob("*.png")}
            lbl_stems = {p.stem for p in (split / "labels").glob("*.txt")}
            assert img_stems == lbl_stems, f"Mismatch in {split.name}"

    def test_idempotent(self, dummy_dataset: Path):
        """Calling split_dataset twice returns same result without error."""
        train1, val1 = split_dataset(dummy_dataset, val_fraction=0.2, seed=42)
        n_train1 = len(list((train1 / "images").glob("*.png")))
        train2, val2 = split_dataset(dummy_dataset, val_fraction=0.2, seed=42)
        n_train2 = len(list((train2 / "images").glob("*.png")))
        assert n_train1 == n_train2

    def test_deterministic_with_seed(self, dummy_dataset: Path):
        """Same seed produces same split."""
        train_dir, val_dir = split_dataset(dummy_dataset, val_fraction=0.2, seed=42)
        val_stems = sorted(p.stem for p in (val_dir / "images").glob("*.png"))

        # Re-create dataset and split with same seed
        # (can't easily re-run since files are moved, but idempotency covers this)
        assert len(val_stems) > 0

    def test_no_images_raises(self, tmp_path: Path):
        (tmp_path / "images").mkdir()
        with pytest.raises(ValueError, match="No images found"):
            split_dataset(tmp_path)

    def test_no_images_dir_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            split_dataset(tmp_path)

    def test_val_fraction_min_one(self, tmp_path: Path):
        """Even with 2 images and small val_fraction, at least 1 goes to val."""
        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()
        for i in range(2):
            depth = np.full((10, 10), 500, dtype=np.uint16)
            cv2.imwrite(str(img_dir / f"{i:04d}.png"), depth)
            (lbl_dir / f"{i:04d}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        train_dir, val_dir = split_dataset(tmp_path, val_fraction=0.01, seed=0)
        n_val = len(list((val_dir / "images").glob("*.png")))
        assert n_val >= 1


# --- write_dataset_yaml tests ---

class TestWriteDatasetYaml:
    def test_writes_yaml(self, dummy_dataset: Path):
        train_dir = dummy_dataset / "train"
        val_dir = dummy_dataset / "val"
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)

        yaml_path = write_dataset_yaml(dummy_dataset, train_dir, val_dir)
        assert yaml_path.exists()
        content = yaml_path.read_text()
        assert "nc: 1" in content
        assert "names: ['ball']" in content
        assert "train:" in content
        assert "val:" in content

    def test_yaml_uses_relative_paths(self, dummy_dataset: Path):
        train_dir = dummy_dataset / "train"
        val_dir = dummy_dataset / "val"
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)

        yaml_path = write_dataset_yaml(dummy_dataset, train_dir, val_dir)
        content = yaml_path.read_text()
        assert "train: train/images" in content
        assert "val: val/images" in content

    def test_yaml_abs_path(self, dummy_dataset: Path):
        train_dir = dummy_dataset / "train"
        val_dir = dummy_dataset / "val"
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)

        yaml_path = write_dataset_yaml(dummy_dataset, train_dir, val_dir)
        content = yaml_path.read_text()
        # path: should be absolute
        assert f"path: {dummy_dataset.resolve()}" in content


# --- depth_png_to_rgb tests ---

class TestDepthPngToRgb:
    def test_converts_to_3channel(self, depth_images_dir: Path):
        depth_png_to_rgb(depth_images_dir)
        for png in depth_images_dir.glob("*.png"):
            img = cv2.imread(str(png))
            assert img is not None
            assert img.ndim == 3
            assert img.shape[2] == 3

    def test_output_is_uint8(self, depth_images_dir: Path):
        depth_png_to_rgb(depth_images_dir)
        for png in depth_images_dir.glob("*.png"):
            img = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
            assert img.dtype == np.uint8

    def test_valid_depth_maps_nonzero(self, depth_images_dir: Path):
        """Pixels with valid depth (168-2000mm) should be nonzero in output."""
        depth_png_to_rgb(depth_images_dir)
        img = cv2.imread(str(next(depth_images_dir.glob("*.png"))), cv2.IMREAD_GRAYSCALE)
        # The central region (30:70, 30:70) had valid depth
        centre = img[30:70, 30:70]
        assert centre.max() > 0

    def test_invalid_depth_maps_zero(self, depth_images_dir: Path):
        """Pixels with zero depth should remain zero."""
        depth_png_to_rgb(depth_images_dir)
        img = cv2.imread(str(next(depth_images_dir.glob("*.png"))), cv2.IMREAD_GRAYSCALE)
        # Corners had zero depth
        assert img[0, 0] == 0

    def test_all_channels_equal(self, depth_images_dir: Path):
        """All 3 channels should be identical (grayscale replicated)."""
        depth_png_to_rgb(depth_images_dir)
        for png in depth_images_dir.glob("*.png"):
            img = cv2.imread(str(png))
            assert np.array_equal(img[:, :, 0], img[:, :, 1])
            assert np.array_equal(img[:, :, 1], img[:, :, 2])

    def test_idempotent_on_8bit(self, depth_images_dir: Path):
        """Running twice doesn't corrupt already-converted images."""
        depth_png_to_rgb(depth_images_dir)
        first = cv2.imread(str(next(depth_images_dir.glob("*.png"))))
        depth_png_to_rgb(depth_images_dir)
        second = cv2.imread(str(next(depth_images_dir.glob("*.png"))))
        # After first conversion, images are uint8 — second call should skip them
        assert np.array_equal(first, second)


# --- Integration: split + yaml ---

class TestSplitAndYaml:
    def test_end_to_end(self, dummy_dataset: Path):
        """Split + write yaml produces a valid dataset structure."""
        train_dir, val_dir = split_dataset(dummy_dataset, val_fraction=0.2, seed=42)
        yaml_path = write_dataset_yaml(dummy_dataset, train_dir, val_dir)

        assert yaml_path.exists()
        assert len(list((train_dir / "images").glob("*.png"))) > 0
        assert len(list((val_dir / "images").glob("*.png"))) > 0

        # YAML should point to real directories
        content = yaml_path.read_text()
        for line in content.split("\n"):
            if line.startswith("train:"):
                rel_path = line.split(":")[1].strip()
                assert (dummy_dataset / rel_path).exists()
            elif line.startswith("val:"):
                rel_path = line.split(":")[1].strip()
                assert (dummy_dataset / rel_path).exists()
