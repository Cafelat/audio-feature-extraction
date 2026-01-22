"""Core data models for audio feature extraction."""

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class AudioData:
    """Audio waveform data.

    Attributes:
        waveform: Audio waveform. Shape: (n_samples,) or (n_channels, n_samples)
        sample_rate: Sampling rate in Hz
        n_channels: Number of audio channels
        duration: Duration in seconds
        metadata: Additional metadata
    """

    waveform: np.ndarray | torch.Tensor
    sample_rate: int
    n_channels: int
    duration: float
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate data shape and type."""
        # Check waveform type
        if isinstance(self.waveform, np.ndarray):
            ndim = self.waveform.ndim
        elif isinstance(self.waveform, torch.Tensor):
            ndim = self.waveform.ndim
        else:
            raise TypeError(
                f"waveform must be np.ndarray or torch.Tensor, got {type(self.waveform)}"
            )

        # Check dimensions
        if ndim not in [1, 2]:
            raise ValueError(f"waveform must be 1D or 2D, got {ndim}D")

        # Check channel count consistency
        if ndim == 1:
            if self.n_channels != 1:
                raise ValueError(
                    f"1D waveform requires n_channels=1, got {self.n_channels}"
                )
        elif ndim == 2:
            actual_channels = self.waveform.shape[0]
            if actual_channels != self.n_channels:
                raise ValueError(
                    f"n_channels mismatch: expected {self.n_channels}, "
                    f"got {actual_channels}"
                )

        # Check sample rate validity
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")

        # Verify duration calculation
        n_samples = self.waveform.shape[-1]
        expected_duration = n_samples / self.sample_rate
        if abs(self.duration - expected_duration) > 1e-3:  # 1ms tolerance
            warnings.warn(
                f"Duration mismatch: provided {self.duration:.3f}s, "
                f"calculated {expected_duration:.3f}s from waveform", stacklevel=2
            )


@dataclass
class SpectrogramData:
    """Spectrogram data from STFT.

    Attributes:
        complex_spec: Complex spectrogram. Shape: (n_frames, n_freq_bins)
        magnitude_db: Magnitude spectrogram in dB scale. Shape: (n_frames, n_freq_bins)
        phase: Phase spectrogram in radians. Shape: (n_frames, n_freq_bins)
        n_fft: FFT size
        hop_length: Hop length between frames
        win_length: Window length
        window: Window function name
        sample_rate: Sampling rate in Hz
        metadata: Additional metadata
    """

    complex_spec: np.ndarray | torch.Tensor
    magnitude_db: np.ndarray | torch.Tensor
    phase: np.ndarray | torch.Tensor
    n_fft: int
    hop_length: int
    win_length: int
    window: str
    sample_rate: int
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate data shape and type."""
        # At least one of complex_spec or (magnitude_db + phase) must be provided
        has_complex = self.complex_spec is not None
        has_mag_phase = self.magnitude_db is not None and self.phase is not None

        if not has_complex and not has_mag_phase:
            raise ValueError(
                "Either complex_spec or both magnitude_db and phase must be provided"
            )

        # Check shape consistency only if both magnitude_db and phase exist
        if self.magnitude_db is not None and self.phase is not None:
            if self.magnitude_db.shape != self.phase.shape:
                raise ValueError(
                    f"magnitude_db and phase shape mismatch: "
                    f"{self.magnitude_db.shape} vs {self.phase.shape}"
                )

        # Check complex_spec shape if it exists and magnitude_db exists
        if self.complex_spec is not None and self.magnitude_db is not None:
            if self.complex_spec.shape != self.magnitude_db.shape:
                raise ValueError(
                    f"complex_spec and magnitude_db shape mismatch: "
                    f"{self.complex_spec.shape} vs {self.magnitude_db.shape}"
                )

        # Verify frequency bin count (use whichever is available)
        expected_freq_bins = self.n_fft // 2 + 1
        if self.magnitude_db is not None:
            actual_freq_bins = self.magnitude_db.shape[1]
        elif self.complex_spec is not None:
            actual_freq_bins = self.complex_spec.shape[1]
        else:
            return  # No data to validate

        if actual_freq_bins != expected_freq_bins:
            raise ValueError(
                f"Frequency bins mismatch: expected {expected_freq_bins} "
                f"(n_fft={self.n_fft}), got {actual_freq_bins}"
            )

        # Validate complex_spec type
        if isinstance(self.complex_spec, np.ndarray):
            if not np.iscomplexobj(self.complex_spec):
                raise TypeError("complex_spec must be complex dtype")
        elif isinstance(self.complex_spec, torch.Tensor):
            if not self.complex_spec.is_complex():
                raise TypeError("complex_spec must be complex dtype")

        # Validate parameters
        if self.n_fft <= 0 or self.hop_length <= 0 or self.win_length <= 0:
            raise ValueError("n_fft, hop_length, win_length must be positive")

        if self.hop_length > self.n_fft:
            raise ValueError(
                f"hop_length ({self.hop_length}) cannot exceed n_fft ({self.n_fft})"
            )

    def to_channels(self) -> np.ndarray | torch.Tensor:
        """Convert to (n_frames, n_freq_bins, 2) shape: [magnitude_db, phase].

        Returns:
            Stacked array with shape (n_frames, n_freq_bins, 2)
        """
        if isinstance(self.magnitude_db, np.ndarray):
            return np.stack([self.magnitude_db, self.phase], axis=-1)
        else:  # torch.Tensor
            return torch.stack([self.magnitude_db, self.phase], dim=-1)

    def save_params(self) -> dict[str, Any]:
        """Get parameters for inverse transform.

        Returns:
            Dictionary containing STFT parameters
        """
        return {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "window": self.window,
            "sample_rate": self.sample_rate,
        }


@dataclass
class MelSpectrogramData:
    """Mel spectrogram data.

    Attributes:
        mel_spec_db: Mel spectrogram in dB scale. Shape: (n_frames, n_mels)
        n_mels: Number of mel filterbanks
        fmin: Minimum frequency in Hz
        fmax: Maximum frequency in Hz
        sample_rate: Sampling rate in Hz
        stft_params: STFT parameters
        metadata: Additional metadata
    """

    mel_spec_db: np.ndarray | torch.Tensor
    n_mels: int
    fmin: float
    fmax: float
    sample_rate: int
    stft_params: dict[str, Any]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate data shape and parameters."""
        # Verify mel bins
        if self.mel_spec_db.ndim != 2:
            raise ValueError(f"mel_spec_db must be 2D, got {self.mel_spec_db.ndim}D")

        actual_n_mels = self.mel_spec_db.shape[1]
        if actual_n_mels != self.n_mels:
            raise ValueError(
                f"n_mels mismatch: expected {self.n_mels}, got {actual_n_mels}"
            )

        # Validate frequency range
        if self.fmin < 0:
            raise ValueError(f"fmin must be non-negative, got {self.fmin}")

        if self.fmax <= self.fmin:
            raise ValueError(
                f"fmax ({self.fmax}) must be greater than fmin ({self.fmin})"
            )

        if self.fmax > self.sample_rate / 2:
            warnings.warn(
                f"fmax ({self.fmax}) exceeds Nyquist frequency ({self.sample_rate / 2})", stacklevel=2
            )


@dataclass
class FeatureData:
    """Generic feature data.

    Attributes:
        features: Feature array
        feature_type: Type of feature (e.g., 'mfcc', 'chroma', 'spectral_centroid')
        shape: Feature shape
        params: Feature extraction parameters
        metadata: Additional metadata
    """

    features: np.ndarray | torch.Tensor
    feature_type: str
    shape: tuple
    params: dict[str, Any]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate feature data."""
        # Verify shape consistency
        if isinstance(self.features, np.ndarray):
            actual_shape = self.features.shape
        elif isinstance(self.features, torch.Tensor):
            actual_shape = tuple(self.features.shape)
        else:
            raise TypeError(
                f"features must be np.ndarray or torch.Tensor, got {type(self.features)}"
            )

        if actual_shape != self.shape:
            raise ValueError(
                f"Shape mismatch: expected {self.shape}, got {actual_shape}"
            )

        # Validate feature type
        if not self.feature_type:
            raise ValueError("feature_type cannot be empty")


@dataclass
class ProcessingState:
    """Processing pipeline state.

    Attributes:
        stage: Current stage ('loaded', 'extracted', 'transformed', 'saved')
        data: Data at current stage
        timestamp: Unix timestamp
        device: Device location ('cpu' or 'cuda:0')
    """

    stage: str
    data: AudioData | SpectrogramData | MelSpectrogramData | FeatureData
    timestamp: float
    device: str

    def __post_init__(self) -> None:
        """Validate processing state."""
        # Validate stage
        valid_stages = {"loaded", "extracted", "transformed", "saved"}
        if self.stage not in valid_stages:
            raise ValueError(f"stage must be one of {valid_stages}, got '{self.stage}'")

        # Validate device
        if not (self.device == "cpu" or self.device.startswith("cuda")):
            raise ValueError(
                f"device must be 'cpu' or 'cuda:*', got '{self.device}'"
            )

        # Validate timestamp
        if self.timestamp < 0:
            raise ValueError(f"timestamp must be non-negative, got {self.timestamp}")

    def describe(self) -> str:
        """Get human-readable description of the state.

        Returns:
            Description string
        """
        data_type = type(self.data).__name__
        return f"Stage: {self.stage} | Data: {data_type} | Device: {self.device}"
