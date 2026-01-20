"""Tests for AudioFileLoader."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from dataset_generator.core.models import AudioData
from dataset_generator.io.audio_loader import AudioFileLoader, AudioLoadError


@pytest.fixture
def temp_wav_file(tmp_path: Path) -> Path:
    """Create temporary WAV file for testing."""
    file_path = tmp_path / "test_audio.wav"
    sample_rate = 16000
    duration = 1.0
    n_samples = int(sample_rate * duration)

    # Generate sine wave
    frequency = 440.0  # A4 note
    t = np.linspace(0, duration, n_samples, dtype=np.float32)
    waveform = np.sin(2 * np.pi * frequency * t)

    sf.write(file_path, waveform, sample_rate)
    return file_path


@pytest.fixture
def temp_stereo_wav_file(tmp_path: Path) -> Path:
    """Create temporary stereo WAV file for testing."""
    file_path = tmp_path / "test_stereo.wav"
    sample_rate = 16000
    duration = 1.0
    n_samples = int(sample_rate * duration)

    # Generate stereo sine waves with different frequencies
    t = np.linspace(0, duration, n_samples, dtype=np.float32)
    left = np.sin(2 * np.pi * 440.0 * t)
    right = np.sin(2 * np.pi * 880.0 * t)
    waveform = np.stack([left, right], axis=1)

    sf.write(file_path, waveform, sample_rate)
    return file_path


@pytest.fixture
def temp_flac_file(tmp_path: Path) -> Path:
    """Create temporary FLAC file for testing."""
    file_path = tmp_path / "test_audio.flac"
    sample_rate = 16000
    duration = 0.5
    n_samples = int(sample_rate * duration)

    # Generate noise
    waveform = np.random.randn(n_samples).astype(np.float32) * 0.1

    sf.write(file_path, waveform, sample_rate, format="FLAC")
    return file_path


class TestAudioFileLoader:
    """Tests for AudioFileLoader class."""

    def test_load_wav_file(self, temp_wav_file: Path) -> None:
        """Test loading WAV file."""
        loader = AudioFileLoader(device="cpu")
        audio = loader.load(str(temp_wav_file))

        assert isinstance(audio, AudioData)
        assert isinstance(audio.waveform, torch.Tensor)
        assert audio.sample_rate == 16000
        assert audio.n_channels == 1
        assert abs(audio.duration - 1.0) < 0.01
        assert audio.waveform.device.type == "cpu"

    def test_load_stereo_wav_file(self, temp_stereo_wav_file: Path) -> None:
        """Test loading stereo WAV file."""
        loader = AudioFileLoader(device="cpu")
        audio = loader.load(str(temp_stereo_wav_file))

        assert isinstance(audio, AudioData)
        assert audio.n_channels == 2
        assert audio.waveform.shape[0] == 2  # (channels, samples)

    def test_load_flac_file(self, temp_flac_file: Path) -> None:
        """Test loading FLAC file."""
        loader = AudioFileLoader(device="cpu")
        audio = loader.load(str(temp_flac_file))

        assert isinstance(audio, AudioData)
        assert audio.sample_rate == 16000
        assert abs(audio.duration - 0.5) < 0.01

    def test_load_with_resampling(self, temp_wav_file: Path) -> None:
        """Test loading with resampling."""
        target_sr = 8000
        loader = AudioFileLoader(target_sr=target_sr, device="cpu")
        audio = loader.load(str(temp_wav_file))

        assert audio.sample_rate == target_sr
        expected_samples = int(1.0 * target_sr)
        actual_samples = len(audio.waveform)
        # Allow some tolerance for resampling
        assert abs(actual_samples - expected_samples) < 100

    def test_load_stereo_as_mono(self, temp_stereo_wav_file: Path) -> None:
        """Test converting stereo to mono."""
        loader = AudioFileLoader(mono=True, device="cpu")
        audio = loader.load(str(temp_stereo_wav_file))

        assert audio.n_channels == 1
        assert audio.waveform.ndim == 1

    def test_load_with_override_parameters(self, temp_wav_file: Path) -> None:
        """Test loading with parameter override."""
        loader = AudioFileLoader(target_sr=16000, mono=False, device="cpu")
        # Override with different parameters
        audio = loader.load(str(temp_wav_file), target_sr=8000, mono=True)

        assert audio.sample_rate == 8000

    def test_load_nonexistent_file(self) -> None:
        """Test loading nonexistent file raises error."""
        loader = AudioFileLoader(device="cpu")

        with pytest.raises(AudioLoadError, match="File not found"):
            loader.load("nonexistent_file.wav")

    def test_load_invalid_file(self, tmp_path: Path) -> None:
        """Test loading invalid file raises error."""
        invalid_file = tmp_path / "invalid.wav"
        invalid_file.write_text("not an audio file")

        loader = AudioFileLoader(device="cpu")

        with pytest.raises(AudioLoadError, match="Failed to load"):
            loader.load(str(invalid_file))

    def test_metadata_populated(self, temp_wav_file: Path) -> None:
        """Test metadata is populated correctly."""
        loader = AudioFileLoader(device="cpu")
        audio = loader.load(str(temp_wav_file))

        assert "file_path" in audio.metadata
        assert audio.metadata["file_path"] == str(temp_wav_file)
        assert audio.metadata["device"] == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_load_to_cuda(self, temp_wav_file: Path) -> None:
        """Test loading directly to CUDA device."""
        loader = AudioFileLoader(device="cuda:0")
        audio = loader.load(str(temp_wav_file))

        assert audio.waveform.device.type == "cuda"
        assert audio.metadata["device"] == "cuda:0"


class TestLoadBatch:
    """Tests for load_batch method."""

    def test_load_batch_multiple_files(
        self, temp_wav_file: Path, temp_flac_file: Path
    ) -> None:
        """Test loading multiple files in batch."""
        loader = AudioFileLoader(device="cpu")
        paths = [str(temp_wav_file), str(temp_flac_file)]

        audios = loader.load_batch(paths)

        assert len(audios) == 2
        assert all(isinstance(a, AudioData) for a in audios)
        assert audios[0].duration > audios[1].duration  # WAV is longer

    def test_load_batch_with_resampling(
        self, temp_wav_file: Path, temp_flac_file: Path
    ) -> None:
        """Test batch loading with resampling."""
        target_sr = 8000
        loader = AudioFileLoader(target_sr=target_sr, device="cpu")
        paths = [str(temp_wav_file), str(temp_flac_file)]

        audios = loader.load_batch(paths)

        assert all(a.sample_rate == target_sr for a in audios)

    def test_load_batch_empty_list(self) -> None:
        """Test loading empty batch."""
        loader = AudioFileLoader(device="cpu")
        audios = loader.load_batch([])

        assert audios == []

    def test_load_batch_with_error(self, temp_wav_file: Path) -> None:
        """Test batch loading with one invalid file."""
        loader = AudioFileLoader(device="cpu")
        paths = [str(temp_wav_file), "nonexistent.wav"]

        with pytest.raises(AudioLoadError):
            loader.load_batch(paths)


class TestResamplingQuality:
    """Tests for resampling quality."""

    def test_resampling_preserves_duration(self, temp_wav_file: Path) -> None:
        """Test that resampling preserves duration."""
        loader_orig = AudioFileLoader(device="cpu")
        loader_resample = AudioFileLoader(target_sr=8000, device="cpu")

        audio_orig = loader_orig.load(str(temp_wav_file))
        audio_resample = loader_resample.load(str(temp_wav_file))

        # Duration should be approximately the same
        assert abs(audio_orig.duration - audio_resample.duration) < 0.01

    def test_downsample_reduces_samples(self, temp_wav_file: Path) -> None:
        """Test that downsampling reduces sample count."""
        loader_orig = AudioFileLoader(device="cpu")
        loader_down = AudioFileLoader(target_sr=8000, device="cpu")

        audio_orig = loader_orig.load(str(temp_wav_file))
        audio_down = loader_down.load(str(temp_wav_file))

        # Downsampled should have fewer samples
        assert len(audio_down.waveform) < len(audio_orig.waveform)

    def test_upsample_increases_samples(self, temp_wav_file: Path) -> None:
        """Test that upsampling increases sample count."""
        loader_orig = AudioFileLoader(device="cpu")
        loader_up = AudioFileLoader(target_sr=48000, device="cpu")

        audio_orig = loader_orig.load(str(temp_wav_file))
        audio_up = loader_up.load(str(temp_wav_file))

        # Upsampled should have more samples
        assert len(audio_up.waveform) > len(audio_orig.waveform)


class TestChannelHandling:
    """Tests for channel handling."""

    def test_stereo_to_mono_averages_channels(
        self, temp_stereo_wav_file: Path
    ) -> None:
        """Test that stereo to mono conversion averages channels."""
        loader_stereo = AudioFileLoader(mono=False, device="cpu")
        loader_mono = AudioFileLoader(mono=True, device="cpu")

        audio_stereo = loader_stereo.load(str(temp_stereo_wav_file))
        audio_mono = loader_mono.load(str(temp_stereo_wav_file))

        assert audio_stereo.n_channels == 2
        assert audio_mono.n_channels == 1

        # Stereo has shape (2, n_samples), mono has shape (n_samples,)
        assert audio_stereo.waveform.ndim == 2
        assert audio_mono.waveform.ndim == 1

    def test_mono_stays_mono(self, temp_wav_file: Path) -> None:
        """Test that mono file stays mono."""
        loader = AudioFileLoader(mono=True, device="cpu")
        audio = loader.load(str(temp_wav_file))

        assert audio.n_channels == 1
        assert audio.waveform.ndim == 1


class TestProtocolCompliance:
    """Tests for AudioLoader protocol compliance."""

    def test_implements_audio_loader_protocol(self) -> None:
        """Test that AudioFileLoader implements AudioLoader protocol."""
        from dataset_generator.core.types import AudioLoader

        loader = AudioFileLoader(device="cpu")
        assert isinstance(loader, AudioLoader)

    def test_has_required_methods(self) -> None:
        """Test that required protocol methods exist."""
        loader = AudioFileLoader(device="cpu")

        assert hasattr(loader, "load")
        assert callable(loader.load)
        assert hasattr(loader, "load_batch")
        assert callable(loader.load_batch)
