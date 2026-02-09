"""Dataset writer for HDF5 format."""

import os
from datetime import datetime
from typing import Optional

import h5py
import numpy as np

from ..core.conversions import TensorConverter
from ..core.models import SpectrogramData
from ..core.types import DatasetFormat


class HDF5DatasetWriter:
    """HDF5 dataset writer with multiple format support.

    Saves spectrogram data in HDF5 format with CNN-compatible shape (time, freq, channels).
    Supports 4 different data formats for model performance comparison.
    """

    def __init__(
        self,
        format: DatasetFormat = DatasetFormat.MAGNITUDE_PHASE_TRIG,
        compression: str = "gzip",
        compression_opts: int = 4,
    ):
        """Initialize HDF5 dataset writer.

        Args:
            format: Dataset storage format
            compression: Compression method ('gzip', 'lzf', None)
            compression_opts: Compression level (0-9 for gzip)
        """
        self.format = format
        self.compression = compression
        self.compression_opts = compression_opts
        self.converter = TensorConverter()

    def write(
        self,
        data: list[SpectrogramData],
        output_path: str,
        split: Optional[str] = None,
    ) -> None:
        """Write dataset to HDF5 file.

        Args:
            data: List of spectrogram data
            output_path: Output HDF5 file path
            split: Data split name ('train', 'val', 'test')
        """
        mode = "a" if os.path.exists(output_path) else "w"

        with h5py.File(output_path, mode) as f:
            # Create split group
            group_name = split if split else "data"
            if group_name in f:
                group = f[group_name]
            else:
                group = f.create_group(group_name)

            # Get starting index for samples
            existing_samples = len(group)

            for i, spec in enumerate(data):
                # Ensure numpy format
                if hasattr(spec.complex_spec, "cpu"):
                    # PyTorch tensor
                    complex_spec = spec.complex_spec.cpu().numpy()
                    magnitude_db = spec.magnitude_db.cpu().numpy()
                    phase = spec.phase.cpu().numpy()
                else:
                    complex_spec = spec.complex_spec
                    magnitude_db = spec.magnitude_db
                    phase = spec.phase

                # Create sample group
                sample_key = f"sample_{existing_samples + i:06d}"
                sample_group = group.create_group(sample_key)

                # Create channel data based on format
                channels_data = self._create_channels(
                    complex_spec, magnitude_db, phase
                )

                # Save CNN-compatible data: (time, freq, channels)
                sample_group.create_dataset(
                    "data",
                    data=channels_data,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )

                # Save metadata
                sample_group.attrs["format"] = self.format.value
                sample_group.attrs["n_channels"] = channels_data.shape[-1]

                # STFT parameters (required for ISTFT)
                sample_group.attrs["n_fft"] = spec.n_fft
                sample_group.attrs["hop_length"] = spec.hop_length
                sample_group.attrs["win_length"] = spec.win_length
                sample_group.attrs["window"] = spec.window
                sample_group.attrs["sample_rate"] = spec.sample_rate

                # Original audio information
                if "original_length" in spec.metadata:
                    sample_group.attrs["original_length"] = spec.metadata[
                        "original_length"
                    ]

                if "preprocessed" in spec.metadata:
                    sample_group.attrs["preprocessed"] = spec.metadata[
                        "preprocessed"
                    ]
                    if spec.metadata.get("target_length"):
                        sample_group.attrs["target_length"] = spec.metadata[
                            "target_length"
                        ]

                # Source file information
                if "source_file" in spec.metadata:
                    sample_group.attrs["source_file"] = spec.metadata[
                        "source_file"
                    ]

                # Data shape
                sample_group.attrs["n_frames"] = channels_data.shape[0]
                sample_group.attrs["n_freq_bins"] = channels_data.shape[1]

            # Global metadata
            group.attrs["n_samples"] = len(group)
            group.attrs["format"] = self.format.value
            group.attrs["created_at"] = datetime.now().isoformat()

            # Common parameters (if all samples have the same values)
            if len(data) > 0:
                group.attrs["common_sample_rate"] = data[0].sample_rate
                group.attrs["common_n_fft"] = data[0].n_fft
                group.attrs["common_hop_length"] = data[0].hop_length

    def _create_channels(
        self,
        complex_spec: np.ndarray,
        magnitude_db: np.ndarray,
        phase: np.ndarray,
    ) -> np.ndarray:
        """Create channel data based on format.

        Args:
            complex_spec: Complex spectrogram
            magnitude_db: Magnitude in dB scale
            phase: Phase in radians

        Returns:
            Channel data with shape (time, freq, channels)
        """
        if self.format == DatasetFormat.COMPLEX:
            # [real, imag] - 2 channels
            channels = np.stack(
                [np.real(complex_spec), np.imag(complex_spec)], axis=-1
            )

        elif self.format == DatasetFormat.MAGNITUDE_PHASE:
            # [mag_db, phase] - 2 channels
            channels = np.stack([magnitude_db, phase], axis=-1)

        elif self.format == DatasetFormat.MAGNITUDE_PHASE_TRIG:
            # [mag_db, cos(phase), sin(phase)] - 3 channels
            channels = np.stack(
                [magnitude_db, np.cos(phase), np.sin(phase)], axis=-1
            )

        elif self.format == DatasetFormat.MAGNITUDE_ONLY:
            # [mag_db] - 1 channel
            channels = magnitude_db[..., np.newaxis]

        else:
            raise ValueError(f"Unknown format: {self.format}")

        return channels
