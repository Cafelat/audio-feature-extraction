"""Tests for Protocol interfaces."""

from typing import Any

import numpy as np

from dataset_generator.core.models import (
    AudioData,
    FeatureData,
    MelSpectrogramData,
    SpectrogramData,
)
from dataset_generator.core.types import (
    AudioLoader,
    DatasetWriter,
    FeatureExtractor,
    InverseTransform,
)


class DummyAudioLoader:
    """Dummy AudioLoader implementation for testing."""

    def load(self, path: str, **kwargs: Any) -> AudioData:
        """Load dummy audio."""
        waveform = np.random.randn(16000)
        return AudioData(
            waveform=waveform,
            sample_rate=16000,
            n_channels=1,
            duration=1.0,
            metadata={},
        )

    def load_batch(self, paths: list[str], **kwargs: Any) -> list[AudioData]:
        """Load batch of dummy audio."""
        return [self.load(path, **kwargs) for path in paths]


class DummyFeatureExtractor:
    """Dummy FeatureExtractor implementation for testing."""

    def extract(self, audio: AudioData) -> SpectrogramData:
        """Extract dummy features."""
        n_fft = 512
        n_frames = 100
        n_freq_bins = n_fft // 2 + 1

        complex_spec = np.random.randn(n_frames, n_freq_bins) + 1j * np.random.randn(
            n_frames, n_freq_bins
        )
        magnitude_db = np.random.randn(n_frames, n_freq_bins)
        phase = np.random.randn(n_frames, n_freq_bins)

        return SpectrogramData(
            complex_spec=complex_spec,
            magnitude_db=magnitude_db,
            phase=phase,
            n_fft=n_fft,
            hop_length=256,
            win_length=512,
            window="hann",
            sample_rate=audio.sample_rate,
            metadata={},
        )

    def get_params(self) -> dict[str, Any]:
        """Get extraction parameters."""
        return {"n_fft": 512, "hop_length": 256, "win_length": 512}


class DummyInverseTransform:
    """Dummy InverseTransform implementation for testing."""

    def reconstruct(
        self, spec: SpectrogramData | MelSpectrogramData
    ) -> AudioData:
        """Reconstruct dummy audio."""
        waveform = np.random.randn(16000)
        return AudioData(
            waveform=waveform,
            sample_rate=spec.sample_rate,
            n_channels=1,
            duration=1.0,
            metadata={},
        )


class DummyDatasetWriter:
    """Dummy DatasetWriter implementation for testing."""

    def write(
        self,
        data: list[SpectrogramData | MelSpectrogramData | FeatureData],
        output_path: str,
        **kwargs: Any,
    ) -> None:
        """Write dummy dataset."""
        pass


class IncompleteAudioLoader:
    """Incomplete AudioLoader (missing load_batch) for testing."""

    def load(self, path: str, **kwargs: Any) -> AudioData:
        """Load dummy audio."""
        waveform = np.random.randn(16000)
        return AudioData(
            waveform=waveform,
            sample_rate=16000,
            n_channels=1,
            duration=1.0,
            metadata={},
        )


class TestAudioLoaderProtocol:
    """Tests for AudioLoader protocol."""

    def test_valid_implementation(self) -> None:
        """Test that valid implementation is recognized."""
        loader = DummyAudioLoader()
        assert isinstance(loader, AudioLoader)

    def test_incomplete_implementation(self) -> None:
        """Test that incomplete implementation is not recognized."""
        loader = IncompleteAudioLoader()
        assert not isinstance(loader, AudioLoader)

    def test_load_method(self) -> None:
        """Test load method works."""
        loader = DummyAudioLoader()
        audio = loader.load("dummy.wav")
        assert isinstance(audio, AudioData)

    def test_load_batch_method(self) -> None:
        """Test load_batch method works."""
        loader = DummyAudioLoader()
        audios = loader.load_batch(["dummy1.wav", "dummy2.wav"])
        assert len(audios) == 2
        assert all(isinstance(a, AudioData) for a in audios)


class TestFeatureExtractorProtocol:
    """Tests for FeatureExtractor protocol."""

    def test_valid_implementation(self) -> None:
        """Test that valid implementation is recognized."""
        extractor = DummyFeatureExtractor()
        assert isinstance(extractor, FeatureExtractor)

    def test_extract_method(self) -> None:
        """Test extract method works."""
        extractor = DummyFeatureExtractor()
        audio = AudioData(
            waveform=np.random.randn(16000),
            sample_rate=16000,
            n_channels=1,
            duration=1.0,
            metadata={},
        )
        spec = extractor.extract(audio)
        assert isinstance(spec, SpectrogramData)

    def test_get_params_method(self) -> None:
        """Test get_params method works."""
        extractor = DummyFeatureExtractor()
        params = extractor.get_params()
        assert isinstance(params, dict)
        assert "n_fft" in params


class TestInverseTransformProtocol:
    """Tests for InverseTransform protocol."""

    def test_valid_implementation(self) -> None:
        """Test that valid implementation is recognized."""
        transform = DummyInverseTransform()
        assert isinstance(transform, InverseTransform)

    def test_reconstruct_method(self) -> None:
        """Test reconstruct method works."""
        transform = DummyInverseTransform()
        n_fft = 512
        n_freq_bins = n_fft // 2 + 1

        spec = SpectrogramData(
            complex_spec=np.random.randn(100, n_freq_bins)
            + 1j * np.random.randn(100, n_freq_bins),
            magnitude_db=np.random.randn(100, n_freq_bins),
            phase=np.random.randn(100, n_freq_bins),
            n_fft=n_fft,
            hop_length=256,
            win_length=512,
            window="hann",
            sample_rate=16000,
            metadata={},
        )

        audio = transform.reconstruct(spec)
        assert isinstance(audio, AudioData)


class TestDatasetWriterProtocol:
    """Tests for DatasetWriter protocol."""

    def test_valid_implementation(self) -> None:
        """Test that valid implementation is recognized."""
        writer = DummyDatasetWriter()
        assert isinstance(writer, DatasetWriter)

    def test_write_method(self) -> None:
        """Test write method works."""
        writer = DummyDatasetWriter()
        n_fft = 512
        n_freq_bins = n_fft // 2 + 1

        data = [
            SpectrogramData(
                complex_spec=np.random.randn(100, n_freq_bins)
                + 1j * np.random.randn(100, n_freq_bins),
                magnitude_db=np.random.randn(100, n_freq_bins),
                phase=np.random.randn(100, n_freq_bins),
                n_fft=n_fft,
                hop_length=256,
                win_length=512,
                window="hann",
                sample_rate=16000,
                metadata={},
            )
        ]

        # Should not raise
        writer.write(data, "dummy.h5")


class TestProtocolComposition:
    """Tests for protocol composition and type checking."""

    def test_multiple_protocols(self) -> None:
        """Test object can implement multiple protocols."""

        class MultiProtocol:
            def load(self, path: str, **kwargs: Any) -> AudioData:
                waveform = np.random.randn(16000)
                return AudioData(
                    waveform=waveform,
                    sample_rate=16000,
                    n_channels=1,
                    duration=1.0,
                    metadata={},
                )

            def load_batch(self, paths: list[str], **kwargs: Any) -> list[AudioData]:
                return [self.load(path, **kwargs) for path in paths]

            def extract(self, audio: AudioData) -> SpectrogramData:
                n_fft = 512
                n_freq_bins = n_fft // 2 + 1
                return SpectrogramData(
                    complex_spec=np.random.randn(100, n_freq_bins)
                    + 1j * np.random.randn(100, n_freq_bins),
                    magnitude_db=np.random.randn(100, n_freq_bins),
                    phase=np.random.randn(100, n_freq_bins),
                    n_fft=n_fft,
                    hop_length=256,
                    win_length=512,
                    window="hann",
                    sample_rate=audio.sample_rate,
                    metadata={},
                )

            def get_params(self) -> dict[str, Any]:
                return {}

        multi = MultiProtocol()
        assert isinstance(multi, AudioLoader)
        assert isinstance(multi, FeatureExtractor)

    def test_protocol_type_hints(self) -> None:
        """Test that protocol type hints work correctly."""

        def process_audio(loader: AudioLoader, path: str) -> AudioData:
            return loader.load(path)

        loader = DummyAudioLoader()
        audio = process_audio(loader, "dummy.wav")
        assert isinstance(audio, AudioData)
