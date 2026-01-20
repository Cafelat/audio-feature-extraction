"""Audio file loading module."""

from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from dataset_generator.core.conversions import TensorConverter
from dataset_generator.core.models import AudioData


class AudioLoadError(Exception):
    """Exception raised when audio loading fails."""

    pass


class AudioFileLoader:
    """Audio file loader with format support and resampling.

    Supports WAV, FLAC, MP3, OGG formats using soundfile and librosa.
    Automatically converts NumPy arrays to PyTorch tensors for internal processing.

    Attributes:
        target_sr: Target sample rate (None to keep original)
        mono: Convert to mono if True
        device: Target device for PyTorch tensors ('cpu' or 'cuda')
    """

    def __init__(
        self,
        target_sr: int | None = None,
        mono: bool = False,
        device: str = "cpu",
    ):
        """Initialize AudioFileLoader.

        Args:
            target_sr: Target sample rate (None to keep original)
            mono: Convert to mono if True
            device: Target device for tensors ('cpu', 'cuda', 'cuda:0', etc.)
        """
        self.target_sr = target_sr
        self.mono = mono
        self.device = device

    def load(self, path: str, **kwargs: Any) -> AudioData:
        """Load audio file.

        Args:
            path: Path to audio file
            **kwargs: Additional parameters (override instance settings)
                - target_sr: Override target sample rate
                - mono: Override mono conversion

        Returns:
            AudioData with waveform as PyTorch tensor

        Raises:
            AudioLoadError: If file loading fails
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise AudioLoadError(f"File not found: {path}")

        # Get override parameters
        target_sr = kwargs.get("target_sr", self.target_sr)
        mono = kwargs.get("mono", self.mono)

        try:
            # Try soundfile first (faster for WAV/FLAC)
            waveform_np, sample_rate = self._load_with_soundfile(path)
        except Exception:
            # Fall back to librosa for MP3/OGG
            try:
                waveform_np, sample_rate = self._load_with_librosa(
                    path, sr=target_sr, mono=mono
                )
            except Exception as e:
                raise AudioLoadError(f"Failed to load {path}: {e}") from e

        # Convert to mono if requested and not already mono
        if mono and waveform_np.ndim == 2:
            waveform_np = np.mean(waveform_np, axis=0)

        # Resample if target sample rate is specified
        if target_sr is not None and sample_rate != target_sr:
            waveform_np = librosa.resample(
                waveform_np,
                orig_sr=sample_rate,
                target_sr=target_sr,
            )
            sample_rate = target_sr

        # Determine channel count
        if waveform_np.ndim == 1:
            n_channels = 1
        else:
            # soundfile: (n_samples, n_channels)
            # We need (n_channels, n_samples) format
            if waveform_np.shape[0] > waveform_np.shape[1]:
                # Likely (n_samples, n_channels), transpose
                waveform_np = waveform_np.T
            n_channels = waveform_np.shape[0]

        # Calculate duration
        n_samples = waveform_np.shape[-1] if waveform_np.ndim > 1 else len(waveform_np)
        duration = n_samples / sample_rate

        # Convert to PyTorch tensor
        waveform_tensor = TensorConverter.to_torch(waveform_np, device=self.device)

        return AudioData(
            waveform=waveform_tensor,
            sample_rate=sample_rate,
            n_channels=n_channels,
            duration=duration,
            metadata={
                "file_path": str(path),
                "device": self.device,
                "original_sr": sample_rate if target_sr is None else None,
            },
        )

    def load_batch(self, paths: list[str], **kwargs: Any) -> list[AudioData]:
        """Load multiple audio files in batch.

        Args:
            paths: List of paths to audio files
            **kwargs: Additional parameters passed to load()

        Returns:
            List of AudioData objects

        Raises:
            AudioLoadError: If any file loading fails
        """
        results = []
        for path in paths:
            try:
                audio = self.load(path, **kwargs)
                results.append(audio)
            except AudioLoadError:
                raise
            except Exception as e:
                raise AudioLoadError(f"Failed to load {path}: {e}") from e

        return results

    def _load_with_soundfile(self, path: str) -> tuple[np.ndarray, int]:
        """Load audio with soundfile (faster for WAV/FLAC).

        Args:
            path: Path to audio file

        Returns:
            Tuple of (waveform, sample_rate)
        """
        waveform, sample_rate = sf.read(path, dtype="float32")
        return waveform, sample_rate

    def _load_with_librosa(
        self, path: str, sr: int | None = None, mono: bool = False
    ) -> tuple[np.ndarray, int]:
        """Load audio with librosa (supports MP3/OGG).

        Args:
            path: Path to audio file
            sr: Target sample rate (None to keep original)
            mono: Convert to mono if True

        Returns:
            Tuple of (waveform, sample_rate)
        """
        waveform, sample_rate = librosa.load(path, sr=sr, mono=mono)
        return waveform, sample_rate
