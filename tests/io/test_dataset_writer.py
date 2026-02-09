"""Tests for HDF5 dataset writer."""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from dataset_generator.core.models import AudioData, SpectrogramData
from dataset_generator.core.types import DatasetFormat
from dataset_generator.io.dataset_writer import HDF5DatasetWriter


class TestHDF5DatasetWriter:
    """Test HDF5 dataset writer."""

    @pytest.fixture
    def sample_spec(self) -> SpectrogramData:
        """Create sample spectrogram data."""
        n_frames, n_freq = 100, 513
        complex_spec = np.random.randn(n_frames, n_freq) + 1j * np.random.randn(
            n_frames, n_freq
        )
        magnitude_db = np.random.randn(n_frames, n_freq) * 20
        phase = np.random.uniform(-np.pi, np.pi, (n_frames, n_freq))

        return SpectrogramData(
            complex_spec=complex_spec,
            magnitude_db=magnitude_db,
            phase=phase,
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            win_length=1024,
            window="hann",
            metadata={"source_file": "test.wav", "original_length": 51200},
        )

    def test_write_magnitude_phase_trig_format(self, sample_spec):
        """Test writing MAGNITUDE_PHASE_TRIG format (default)."""
        writer = HDF5DatasetWriter(format=DatasetFormat.MAGNITUDE_PHASE_TRIG)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            output_path = f.name

        try:
            # Write dataset
            writer.write([sample_spec], output_path, split="train")

            # Verify file exists and data is correct
            with h5py.File(output_path, "r") as hf:
                assert "train" in hf
                train_group = hf["train"]

                assert "sample_000000" in train_group
                sample = train_group["sample_000000"]

                # Check data shape: (time, freq, 3)
                assert "data" in sample
                data = sample["data"][:]
                assert data.shape == (100, 513, 3)  # [mag_db, cos, sin]

                # Check metadata
                assert sample.attrs["format"] == "magnitude_phase_trig"
                assert sample.attrs["n_channels"] == 3
                assert sample.attrs["n_fft"] == 1024
                assert sample.attrs["hop_length"] == 512
                assert sample.attrs["sample_rate"] == 22050
                assert sample.attrs["original_length"] == 51200

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_write_complex_format(self, sample_spec):
        """Test writing COMPLEX format."""
        writer = HDF5DatasetWriter(format=DatasetFormat.COMPLEX)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            output_path = f.name

        try:
            writer.write([sample_spec], output_path, split="train")

            with h5py.File(output_path, "r") as hf:
                data = hf["train/sample_000000/data"][:]
                assert data.shape == (100, 513, 2)  # [real, imag]
                assert hf["train/sample_000000"].attrs["n_channels"] == 2

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_write_magnitude_phase_format(self, sample_spec):
        """Test writing MAGNITUDE_PHASE format."""
        writer = HDF5DatasetWriter(format=DatasetFormat.MAGNITUDE_PHASE)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            output_path = f.name

        try:
            writer.write([sample_spec], output_path, split="train")

            with h5py.File(output_path, "r") as hf:
                data = hf["train/sample_000000/data"][:]
                assert data.shape == (100, 513, 2)  # [mag_db, phase]
                assert hf["train/sample_000000"].attrs["n_channels"] == 2

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_write_magnitude_only_format(self, sample_spec):
        """Test writing MAGNITUDE_ONLY format."""
        writer = HDF5DatasetWriter(format=DatasetFormat.MAGNITUDE_ONLY)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            output_path = f.name

        try:
            writer.write([sample_spec], output_path, split="train")

            with h5py.File(output_path, "r") as hf:
                data = hf["train/sample_000000/data"][:]
                assert data.shape == (100, 513, 1)  # [mag_db]
                assert hf["train/sample_000000"].attrs["n_channels"] == 1

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_write_with_compression(self, sample_spec):
        """Test writing with compression."""
        writer = HDF5DatasetWriter(compression="gzip", compression_opts=4)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            output_path = f.name

        try:
            writer.write([sample_spec], output_path)

            with h5py.File(output_path, "r") as hf:
                dataset = hf["data/sample_000000/data"]
                assert dataset.compression == "gzip"
                assert dataset.compression_opts == 4

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_write_multiple_samples(self, sample_spec):
        """Test writing multiple samples."""
        writer = HDF5DatasetWriter()

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            output_path = f.name

        try:
            # Write 3 samples
            writer.write([sample_spec, sample_spec, sample_spec], output_path)

            with h5py.File(output_path, "r") as hf:
                assert "sample_000000" in hf["data"]
                assert "sample_000001" in hf["data"]
                assert "sample_000002" in hf["data"]
                assert hf["data"].attrs["n_samples"] == 3

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_write_multiple_splits(self, sample_spec):
        """Test writing multiple data splits."""
        writer = HDF5DatasetWriter()

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            output_path = f.name

        try:
            # Write train split
            writer.write([sample_spec], output_path, split="train")
            # Write val split
            writer.write([sample_spec], output_path, split="val")
            # Write test split
            writer.write([sample_spec], output_path, split="test")

            with h5py.File(output_path, "r") as hf:
                assert "train" in hf
                assert "val" in hf
                assert "test" in hf
                assert len(hf["train"]) > 0
                assert len(hf["val"]) > 0
                assert len(hf["test"]) > 0

        finally:
            Path(output_path).unlink(missing_ok=True)
