"""Type definitions and Protocol interfaces."""

from typing import Any, Protocol, runtime_checkable

from dataset_generator.core.models import (
    AudioData,
    FeatureData,
    MelSpectrogramData,
    SpectrogramData,
)


@runtime_checkable
class AudioLoader(Protocol):
    """Audio loading interface.

    This protocol defines the interface for audio file loading implementations.
    Any class implementing load() and load_batch() methods can be used as
    an AudioLoader.
    """

    def load(self, path: str, **kwargs: Any) -> AudioData:
        """Load audio file.

        Args:
            path: Path to audio file
            **kwargs: Additional loading parameters

        Returns:
            Loaded audio data
        """
        ...

    def load_batch(self, paths: list[str], **kwargs: Any) -> list[AudioData]:
        """Load multiple audio files in batch.

        Args:
            paths: List of paths to audio files
            **kwargs: Additional loading parameters

        Returns:
            List of loaded audio data
        """
        ...


@runtime_checkable
class FeatureExtractor(Protocol):
    """Feature extraction interface.

    This protocol defines the interface for feature extraction implementations
    such as STFT, Mel spectrogram, MFCC, etc.
    """

    def extract(
        self, audio: AudioData
    ) -> SpectrogramData | MelSpectrogramData | FeatureData:
        """Extract features from audio.

        Args:
            audio: Input audio data

        Returns:
            Extracted features (spectrogram, mel spectrogram, or generic features)
        """
        ...

    def get_params(self) -> dict[str, Any]:
        """Get extraction parameters.

        Returns:
            Dictionary of extraction parameters
        """
        ...


@runtime_checkable
class InverseTransform(Protocol):
    """Inverse transform interface.

    This protocol defines the interface for inverse transforms such as
    ISTFT and Griffin-Lim algorithm.
    """

    def reconstruct(
        self, spec: SpectrogramData | MelSpectrogramData
    ) -> AudioData:
        """Reconstruct audio from spectrogram.

        Args:
            spec: Input spectrogram data

        Returns:
            Reconstructed audio data
        """
        ...


@runtime_checkable
class DatasetWriter(Protocol):
    """Dataset writing interface.

    This protocol defines the interface for dataset writers such as
    HDF5 writer, TFRecord writer, etc.
    """

    def write(
        self,
        data: list[SpectrogramData | MelSpectrogramData | FeatureData],
        output_path: str,
        **kwargs: Any,
    ) -> None:
        """Write dataset to file.

        Args:
            data: List of feature data to write
            output_path: Output file path
            **kwargs: Additional writing parameters
        """
        ...
