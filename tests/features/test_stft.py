"""Tests for STFT extractor."""

import numpy as np
import pytest
import torch

from dataset_generator.core.models import AudioData, SpectrogramData
from dataset_generator.features.stft import STFTExtractor


@pytest.fixture
def sample_audio(sample_rate: int) -> AudioData:
    """Create sample audio with sine wave."""
    duration = 1.0
    frequency = 440.0  # A4 note
    n_samples = int(sample_rate * duration)

    t = np.linspace(0, duration, n_samples, dtype=np.float32)
    waveform = np.sin(2 * np.pi * frequency * t)

    return AudioData(
        waveform=torch.from_numpy(waveform),
        sample_rate=sample_rate,
        n_channels=1,
        duration=duration,
        metadata={"file_path": "test.wav"},
    )


@pytest.fixture
def stereo_audio(sample_rate: int) -> AudioData:
    """Create stereo audio."""
    duration = 1.0
    n_samples = int(sample_rate * duration)

    t = np.linspace(0, duration, n_samples, dtype=np.float32)
    left = np.sin(2 * np.pi * 440.0 * t)
    right = np.sin(2 * np.pi * 880.0 * t)
    waveform = np.stack([left, right], axis=0)

    return AudioData(
        waveform=torch.from_numpy(waveform),
        sample_rate=sample_rate,
        n_channels=2,
        duration=duration,
        metadata={},
    )


class TestSTFTExtractor:
    """Tests for STFTExtractor class."""

    def test_initialization(self) -> None:
        """Test extractor initialization."""
        extractor = STFTExtractor(
            n_fft=2048, hop_length=512, win_length=2048, window="hann", device="cpu"
        )

        assert extractor.n_fft == 2048
        assert extractor.hop_length == 512
        assert extractor.win_length == 2048
        assert extractor.window == "hann"
        assert extractor.device == "cpu"

    def test_default_win_length(self) -> None:
        """Test that win_length defaults to n_fft."""
        extractor = STFTExtractor(n_fft=2048, hop_length=512)

        assert extractor.win_length == 2048

    def test_extract_basic(self, sample_audio: AudioData) -> None:
        """Test basic STFT extraction."""
        extractor = STFTExtractor(n_fft=512, hop_length=256, device="cpu")
        spec = extractor.extract(sample_audio)

        assert isinstance(spec, SpectrogramData)
        assert spec.n_fft == 512
        assert spec.hop_length == 256
        assert spec.sample_rate == sample_audio.sample_rate

    def test_frequency_bins(self, sample_audio: AudioData) -> None:
        """Test frequency bin count."""
        n_fft = 512
        extractor = STFTExtractor(n_fft=n_fft, hop_length=256, device="cpu")
        spec = extractor.extract(sample_audio)

        expected_freq_bins = n_fft // 2 + 1
        assert spec.magnitude_db.shape[1] == expected_freq_bins
        assert spec.phase.shape[1] == expected_freq_bins
        assert spec.complex_spec.shape[1] == expected_freq_bins

    def test_complex_spectrogram_type(self, sample_audio: AudioData) -> None:
        """Test that complex spectrogram is complex type."""
        extractor = STFTExtractor(n_fft=512, hop_length=256, device="cpu")
        spec = extractor.extract(sample_audio)

        assert spec.complex_spec.is_complex()
        assert spec.complex_spec.dtype in [torch.complex64, torch.complex128]

    def test_magnitude_and_phase_real(self, sample_audio: AudioData) -> None:
        """Test that magnitude and phase are real values."""
        extractor = STFTExtractor(n_fft=512, hop_length=256, device="cpu")
        spec = extractor.extract(sample_audio)

        assert not spec.magnitude_db.is_complex()
        assert not spec.phase.is_complex()

    def test_db_scale_range(self, sample_audio: AudioData) -> None:
        """Test that dB values are in reasonable range."""
        extractor = STFTExtractor(n_fft=512, hop_length=256, device="cpu")
        spec = extractor.extract(sample_audio)

        # dB values should be finite
        assert torch.all(torch.isfinite(spec.magnitude_db))
        # Should be within reasonable range (can be positive if amplitude > ref)
        assert spec.magnitude_db.min() >= -100

    def test_phase_range(self, sample_audio: AudioData) -> None:
        """Test that phase is in [-π, π] range."""
        extractor = STFTExtractor(n_fft=512, hop_length=256, device="cpu")
        spec = extractor.extract(sample_audio)

        assert spec.phase.min() >= -np.pi
        assert spec.phase.max() <= np.pi

    def test_sine_wave_peak_frequency(self, sample_rate: int) -> None:
        """Test that sine wave produces peak at correct frequency."""
        frequency = 440.0
        duration = 1.0
        n_samples = int(sample_rate * duration)

        t = np.linspace(0, duration, n_samples, dtype=np.float32)
        waveform = np.sin(2 * np.pi * frequency * t)

        audio = AudioData(
            waveform=torch.from_numpy(waveform),
            sample_rate=sample_rate,
            n_channels=1,
            duration=duration,
            metadata={},
        )

        n_fft = 2048
        extractor = STFTExtractor(n_fft=n_fft, hop_length=512, device="cpu")
        spec = extractor.extract(audio)

        # Find peak frequency
        magnitude = torch.abs(spec.complex_spec)
        avg_magnitude = magnitude.mean(dim=0)  # Average over time
        peak_bin = torch.argmax(avg_magnitude).item()
        peak_freq = peak_bin * sample_rate / n_fft

        # Should be close to 440 Hz (within one bin)
        freq_resolution = sample_rate / n_fft
        assert abs(peak_freq - frequency) < freq_resolution * 2

    def test_stereo_audio_uses_first_channel(self, stereo_audio: AudioData) -> None:
        """Test that stereo audio uses first channel."""
        extractor = STFTExtractor(n_fft=512, hop_length=256, device="cpu")
        spec = extractor.extract(stereo_audio)

        assert isinstance(spec, SpectrogramData)
        # Should extract successfully from first channel

    def test_metadata_populated(self, sample_audio: AudioData) -> None:
        """Test that metadata is populated."""
        extractor = STFTExtractor(n_fft=512, hop_length=256, device="cpu")
        spec = extractor.extract(sample_audio)

        assert "source_file" in spec.metadata
        assert spec.metadata["source_file"] == "test.wav"
        assert spec.metadata["device"] == "cpu"

    def test_get_params(self) -> None:
        """Test get_params method."""
        extractor = STFTExtractor(
            n_fft=2048, hop_length=512, win_length=2048, window="hann", center=True
        )

        params = extractor.get_params()

        assert params["n_fft"] == 2048
        assert params["hop_length"] == 512
        assert params["win_length"] == 2048
        assert params["window"] == "hann"
        assert params["center"] is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_extract_on_cuda(self, sample_audio: AudioData) -> None:
        """Test extraction on CUDA device."""
        extractor = STFTExtractor(n_fft=512, hop_length=256, device="cuda:0")
        spec = extractor.extract(sample_audio)

        assert spec.complex_spec.device.type == "cuda"
        assert spec.magnitude_db.device.type == "cuda"
        assert spec.phase.device.type == "cuda"


class TestWindowFunctions:
    """Tests for different window functions."""

    def test_hann_window(self, sample_audio: AudioData) -> None:
        """Test STFT with Hann window."""
        extractor = STFTExtractor(n_fft=512, hop_length=256, window="hann")
        spec = extractor.extract(sample_audio)

        assert isinstance(spec, SpectrogramData)

    def test_hamming_window(self, sample_audio: AudioData) -> None:
        """Test STFT with Hamming window."""
        extractor = STFTExtractor(n_fft=512, hop_length=256, window="hamming")
        spec = extractor.extract(sample_audio)

        assert isinstance(spec, SpectrogramData)

    def test_blackman_window(self, sample_audio: AudioData) -> None:
        """Test STFT with Blackman window."""
        extractor = STFTExtractor(n_fft=512, hop_length=256, window="blackman")
        spec = extractor.extract(sample_audio)

        assert isinstance(spec, SpectrogramData)

    def test_unknown_window_defaults_to_hann(self, sample_audio: AudioData) -> None:
        """Test that unknown window defaults to Hann."""
        extractor = STFTExtractor(n_fft=512, hop_length=256, window="unknown")
        spec = extractor.extract(sample_audio)

        assert isinstance(spec, SpectrogramData)


class TestDbConversion:
    """Tests for dB conversion."""

    def test_to_db_basic(self) -> None:
        """Test basic dB conversion."""
        extractor = STFTExtractor()
        magnitude = torch.tensor([1.0, 0.1, 0.01])

        db = extractor._to_db(magnitude)

        # 20 * log10(1.0) = 0
        assert abs(db[0].item() - 0.0) < 0.01
        # 20 * log10(0.1) = -20
        assert abs(db[1].item() - (-20.0)) < 0.01
        # 20 * log10(0.01) = -40
        assert abs(db[2].item() - (-40.0)) < 0.01

    def test_to_db_clamps_minimum(self) -> None:
        """Test that very small values are clamped."""
        extractor = STFTExtractor()
        magnitude = torch.tensor([1e-20, 1e-30])

        db = extractor._to_db(magnitude, amin=1e-10)

        # Should clamp to amin before log
        assert torch.all(torch.isfinite(db))

    def test_to_db_top_db_clamping(self) -> None:
        """Test top_db clamping."""
        extractor = STFTExtractor()
        magnitude = torch.tensor([1.0, 0.001])  # 0 dB and -60 dB

        db = extractor._to_db(magnitude, top_db=40.0)

        # Maximum should be 0
        assert abs(db.max().item() - 0.0) < 0.01
        # Minimum should be clamped to -40
        assert db.min().item() >= -40.0


class TestDeviceTransfer:
    """Tests for device transfer."""

    def test_to_device_numpy_array(self) -> None:
        """Test transferring NumPy array to device."""
        extractor = STFTExtractor(device="cpu")
        array = np.random.randn(100, 50).astype(np.float32)

        tensor = extractor._to_device(array)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == "cpu"

    def test_to_device_torch_tensor(self) -> None:
        """Test transferring PyTorch tensor to device."""
        extractor = STFTExtractor(device="cpu")
        tensor_in = torch.randn(100, 50)

        tensor_out = extractor._to_device(tensor_in)

        assert isinstance(tensor_out, torch.Tensor)
        assert tensor_out.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_device_cuda(self) -> None:
        """Test transferring to CUDA device."""
        extractor = STFTExtractor(device="cuda:0")
        tensor_in = torch.randn(100, 50)

        tensor_out = extractor._to_device(tensor_in)

        assert tensor_out.device.type == "cuda"


class TestProtocolCompliance:
    """Tests for FeatureExtractor protocol compliance."""

    def test_implements_feature_extractor_protocol(self) -> None:
        """Test that STFTExtractor implements FeatureExtractor protocol."""
        from dataset_generator.core.types import FeatureExtractor

        extractor = STFTExtractor()
        assert isinstance(extractor, FeatureExtractor)

    def test_has_required_methods(self) -> None:
        """Test that required protocol methods exist."""
        extractor = STFTExtractor()

        assert hasattr(extractor, "extract")
        assert callable(extractor.extract)
        assert hasattr(extractor, "get_params")
        assert callable(extractor.get_params)
