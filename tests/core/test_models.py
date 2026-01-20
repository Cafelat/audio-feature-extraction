"""Tests for core data models."""

import time

import numpy as np
import pytest
import torch

from dataset_generator.core.models import (
    AudioData,
    FeatureData,
    MelSpectrogramData,
    ProcessingState,
    SpectrogramData,
)


class TestAudioData:
    """Tests for AudioData class."""

    def test_valid_mono_numpy(self, sample_rate: int) -> None:
        """Test valid mono audio with NumPy array."""
        waveform = np.random.randn(16000)
        duration = len(waveform) / sample_rate

        audio = AudioData(
            waveform=waveform,
            sample_rate=sample_rate,
            n_channels=1,
            duration=duration,
            metadata={},
        )

        assert audio.sample_rate == sample_rate
        assert audio.n_channels == 1
        assert isinstance(audio.waveform, np.ndarray)

    def test_valid_stereo_numpy(self, sample_rate: int) -> None:
        """Test valid stereo audio with NumPy array."""
        waveform = np.random.randn(2, 16000)
        duration = waveform.shape[1] / sample_rate

        audio = AudioData(
            waveform=waveform,
            sample_rate=sample_rate,
            n_channels=2,
            duration=duration,
            metadata={},
        )

        assert audio.n_channels == 2
        assert audio.waveform.shape[0] == 2

    def test_valid_mono_torch(self, sample_rate: int) -> None:
        """Test valid mono audio with PyTorch tensor."""
        waveform = torch.randn(16000)
        duration = len(waveform) / sample_rate

        audio = AudioData(
            waveform=waveform,
            sample_rate=sample_rate,
            n_channels=1,
            duration=duration,
            metadata={},
        )

        assert isinstance(audio.waveform, torch.Tensor)
        assert audio.n_channels == 1

    def test_invalid_waveform_type(self, sample_rate: int) -> None:
        """Test invalid waveform type."""
        with pytest.raises(TypeError, match="waveform must be np.ndarray or torch.Tensor"):
            AudioData(
                waveform=[1, 2, 3],  # type: ignore
                sample_rate=sample_rate,
                n_channels=1,
                duration=1.0,
                metadata={},
            )

    def test_invalid_dimensions(self, sample_rate: int) -> None:
        """Test invalid waveform dimensions."""
        waveform = np.random.randn(2, 2, 16000)  # 3D

        with pytest.raises(ValueError, match="waveform must be 1D or 2D"):
            AudioData(
                waveform=waveform,
                sample_rate=sample_rate,
                n_channels=2,
                duration=1.0,
                metadata={},
            )

    def test_channel_mismatch_mono(self, sample_rate: int) -> None:
        """Test channel count mismatch for mono audio."""
        waveform = np.random.randn(16000)

        with pytest.raises(ValueError, match="1D waveform requires n_channels=1"):
            AudioData(
                waveform=waveform,
                sample_rate=sample_rate,
                n_channels=2,
                duration=1.0,
                metadata={},
            )

    def test_channel_mismatch_stereo(self, sample_rate: int) -> None:
        """Test channel count mismatch for stereo audio."""
        waveform = np.random.randn(2, 16000)

        with pytest.raises(ValueError, match="n_channels mismatch"):
            AudioData(
                waveform=waveform,
                sample_rate=sample_rate,
                n_channels=1,
                duration=1.0,
                metadata={},
            )

    def test_invalid_sample_rate(self) -> None:
        """Test invalid sample rate."""
        waveform = np.random.randn(16000)

        with pytest.raises(ValueError, match="sample_rate must be positive"):
            AudioData(
                waveform=waveform,
                sample_rate=0,
                n_channels=1,
                duration=1.0,
                metadata={},
            )

    def test_duration_mismatch_warning(self, sample_rate: int) -> None:
        """Test warning for duration mismatch."""
        waveform = np.random.randn(16000)
        incorrect_duration = 2.0  # Should be 1.0

        with pytest.warns(UserWarning, match="Duration mismatch"):
            AudioData(
                waveform=waveform,
                sample_rate=sample_rate,
                n_channels=1,
                duration=incorrect_duration,
                metadata={},
            )


class TestSpectrogramData:
    """Tests for SpectrogramData class."""

    def test_valid_spectrogram_numpy(self, sample_rate: int) -> None:
        """Test valid spectrogram with NumPy arrays."""
        n_fft = 512
        n_frames = 100
        n_freq_bins = n_fft // 2 + 1

        complex_spec = np.random.randn(n_frames, n_freq_bins) + 1j * np.random.randn(
            n_frames, n_freq_bins
        )
        magnitude_db = np.random.randn(n_frames, n_freq_bins)
        phase = np.random.randn(n_frames, n_freq_bins)

        spec = SpectrogramData(
            complex_spec=complex_spec,
            magnitude_db=magnitude_db,
            phase=phase,
            n_fft=n_fft,
            hop_length=256,
            win_length=512,
            window="hann",
            sample_rate=sample_rate,
            metadata={},
        )

        assert spec.n_fft == n_fft
        assert spec.complex_spec.shape == (n_frames, n_freq_bins)

    def test_valid_spectrogram_torch(self, sample_rate: int) -> None:
        """Test valid spectrogram with PyTorch tensors."""
        n_fft = 512
        n_frames = 100
        n_freq_bins = n_fft // 2 + 1

        complex_spec = torch.randn(n_frames, n_freq_bins, dtype=torch.complex64)
        magnitude_db = torch.randn(n_frames, n_freq_bins)
        phase = torch.randn(n_frames, n_freq_bins)

        spec = SpectrogramData(
            complex_spec=complex_spec,
            magnitude_db=magnitude_db,
            phase=phase,
            n_fft=n_fft,
            hop_length=256,
            win_length=512,
            window="hann",
            sample_rate=sample_rate,
            metadata={},
        )

        assert isinstance(spec.complex_spec, torch.Tensor)

    def test_shape_mismatch_magnitude_phase(self, sample_rate: int) -> None:
        """Test shape mismatch between magnitude and phase."""
        n_fft = 512
        n_freq_bins = n_fft // 2 + 1

        complex_spec = np.random.randn(100, n_freq_bins) + 1j * np.random.randn(
            100, n_freq_bins
        )
        magnitude_db = np.random.randn(100, n_freq_bins)
        phase = np.random.randn(90, n_freq_bins)  # Different n_frames

        with pytest.raises(ValueError, match="magnitude_db and phase shape mismatch"):
            SpectrogramData(
                complex_spec=complex_spec,
                magnitude_db=magnitude_db,
                phase=phase,
                n_fft=n_fft,
                hop_length=256,
                win_length=512,
                window="hann",
                sample_rate=sample_rate,
                metadata={},
            )

    def test_frequency_bins_mismatch(self, sample_rate: int) -> None:
        """Test frequency bins mismatch."""
        n_fft = 512
        wrong_freq_bins = 200

        complex_spec = np.random.randn(100, wrong_freq_bins) + 1j * np.random.randn(
            100, wrong_freq_bins
        )
        magnitude_db = np.random.randn(100, wrong_freq_bins)
        phase = np.random.randn(100, wrong_freq_bins)

        with pytest.raises(ValueError, match="Frequency bins mismatch"):
            SpectrogramData(
                complex_spec=complex_spec,
                magnitude_db=magnitude_db,
                phase=phase,
                n_fft=n_fft,
                hop_length=256,
                win_length=512,
                window="hann",
                sample_rate=sample_rate,
                metadata={},
            )

    def test_non_complex_spec_numpy(self, sample_rate: int) -> None:
        """Test non-complex spectrogram with NumPy."""
        n_fft = 512
        n_freq_bins = n_fft // 2 + 1

        complex_spec = np.random.randn(100, n_freq_bins)  # Real, not complex
        magnitude_db = np.random.randn(100, n_freq_bins)
        phase = np.random.randn(100, n_freq_bins)

        with pytest.raises(TypeError, match="complex_spec must be complex dtype"):
            SpectrogramData(
                complex_spec=complex_spec,
                magnitude_db=magnitude_db,
                phase=phase,
                n_fft=n_fft,
                hop_length=256,
                win_length=512,
                window="hann",
                sample_rate=sample_rate,
                metadata={},
            )

    def test_non_complex_spec_torch(self, sample_rate: int) -> None:
        """Test non-complex spectrogram with PyTorch."""
        n_fft = 512
        n_freq_bins = n_fft // 2 + 1

        complex_spec = torch.randn(100, n_freq_bins)  # Real, not complex
        magnitude_db = torch.randn(100, n_freq_bins)
        phase = torch.randn(100, n_freq_bins)

        with pytest.raises(TypeError, match="complex_spec must be complex dtype"):
            SpectrogramData(
                complex_spec=complex_spec,
                magnitude_db=magnitude_db,
                phase=phase,
                n_fft=n_fft,
                hop_length=256,
                win_length=512,
                window="hann",
                sample_rate=sample_rate,
                metadata={},
            )

    def test_invalid_hop_length(self, sample_rate: int) -> None:
        """Test invalid hop length."""
        n_fft = 512
        n_freq_bins = n_fft // 2 + 1

        complex_spec = np.random.randn(100, n_freq_bins) + 1j * np.random.randn(
            100, n_freq_bins
        )
        magnitude_db = np.random.randn(100, n_freq_bins)
        phase = np.random.randn(100, n_freq_bins)

        with pytest.raises(
            ValueError, match="hop_length .* cannot exceed n_fft"
        ):
            SpectrogramData(
                complex_spec=complex_spec,
                magnitude_db=magnitude_db,
                phase=phase,
                n_fft=n_fft,
                hop_length=1024,  # Greater than n_fft
                win_length=512,
                window="hann",
                sample_rate=sample_rate,
                metadata={},
            )

    def test_to_channels_numpy(self, sample_rate: int) -> None:
        """Test to_channels method with NumPy."""
        n_fft = 512
        n_frames = 100
        n_freq_bins = n_fft // 2 + 1

        complex_spec = np.random.randn(n_frames, n_freq_bins) + 1j * np.random.randn(
            n_frames, n_freq_bins
        )
        magnitude_db = np.random.randn(n_frames, n_freq_bins)
        phase = np.random.randn(n_frames, n_freq_bins)

        spec = SpectrogramData(
            complex_spec=complex_spec,
            magnitude_db=magnitude_db,
            phase=phase,
            n_fft=n_fft,
            hop_length=256,
            win_length=512,
            window="hann",
            sample_rate=sample_rate,
            metadata={},
        )

        channels = spec.to_channels()
        assert channels.shape == (n_frames, n_freq_bins, 2)
        assert isinstance(channels, np.ndarray)

    def test_to_channels_torch(self, sample_rate: int) -> None:
        """Test to_channels method with PyTorch."""
        n_fft = 512
        n_frames = 100
        n_freq_bins = n_fft // 2 + 1

        complex_spec = torch.randn(n_frames, n_freq_bins, dtype=torch.complex64)
        magnitude_db = torch.randn(n_frames, n_freq_bins)
        phase = torch.randn(n_frames, n_freq_bins)

        spec = SpectrogramData(
            complex_spec=complex_spec,
            magnitude_db=magnitude_db,
            phase=phase,
            n_fft=n_fft,
            hop_length=256,
            win_length=512,
            window="hann",
            sample_rate=sample_rate,
            metadata={},
        )

        channels = spec.to_channels()
        assert channels.shape == (n_frames, n_freq_bins, 2)
        assert isinstance(channels, torch.Tensor)

    def test_save_params(self, sample_rate: int) -> None:
        """Test save_params method."""
        n_fft = 512
        hop_length = 256
        win_length = 512
        window = "hann"
        n_freq_bins = n_fft // 2 + 1

        complex_spec = np.random.randn(100, n_freq_bins) + 1j * np.random.randn(
            100, n_freq_bins
        )
        magnitude_db = np.random.randn(100, n_freq_bins)
        phase = np.random.randn(100, n_freq_bins)

        spec = SpectrogramData(
            complex_spec=complex_spec,
            magnitude_db=magnitude_db,
            phase=phase,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            sample_rate=sample_rate,
            metadata={},
        )

        params = spec.save_params()
        assert params["n_fft"] == n_fft
        assert params["hop_length"] == hop_length
        assert params["win_length"] == win_length
        assert params["window"] == window
        assert params["sample_rate"] == sample_rate


class TestMelSpectrogramData:
    """Tests for MelSpectrogramData class."""

    def test_valid_mel_spectrogram(self, sample_rate: int) -> None:
        """Test valid mel spectrogram."""
        n_mels = 80
        mel_spec_db = np.random.randn(100, n_mels)

        mel_spec = MelSpectrogramData(
            mel_spec_db=mel_spec_db,
            n_mels=n_mels,
            fmin=0.0,
            fmax=8000.0,
            sample_rate=sample_rate,
            stft_params={"n_fft": 512, "hop_length": 256},
            metadata={},
        )

        assert mel_spec.n_mels == n_mels

    def test_invalid_dimensions(self, sample_rate: int) -> None:
        """Test invalid mel spectrogram dimensions."""
        mel_spec_db = np.random.randn(100)  # 1D

        with pytest.raises(ValueError, match="mel_spec_db must be 2D"):
            MelSpectrogramData(
                mel_spec_db=mel_spec_db,
                n_mels=80,
                fmin=0.0,
                fmax=8000.0,
                sample_rate=sample_rate,
                stft_params={},
                metadata={},
            )

    def test_n_mels_mismatch(self, sample_rate: int) -> None:
        """Test n_mels mismatch."""
        mel_spec_db = np.random.randn(100, 80)

        with pytest.raises(ValueError, match="n_mels mismatch"):
            MelSpectrogramData(
                mel_spec_db=mel_spec_db,
                n_mels=64,  # Wrong value
                fmin=0.0,
                fmax=8000.0,
                sample_rate=sample_rate,
                stft_params={},
                metadata={},
            )

    def test_invalid_fmin(self, sample_rate: int) -> None:
        """Test invalid fmin."""
        mel_spec_db = np.random.randn(100, 80)

        with pytest.raises(ValueError, match="fmin must be non-negative"):
            MelSpectrogramData(
                mel_spec_db=mel_spec_db,
                n_mels=80,
                fmin=-1.0,
                fmax=8000.0,
                sample_rate=sample_rate,
                stft_params={},
                metadata={},
            )

    def test_invalid_fmax(self, sample_rate: int) -> None:
        """Test invalid fmax."""
        mel_spec_db = np.random.randn(100, 80)

        with pytest.raises(ValueError, match="fmax .* must be greater than fmin"):
            MelSpectrogramData(
                mel_spec_db=mel_spec_db,
                n_mels=80,
                fmin=8000.0,
                fmax=4000.0,  # Less than fmin
                sample_rate=sample_rate,
                stft_params={},
                metadata={},
            )

    def test_fmax_exceeds_nyquist_warning(self, sample_rate: int) -> None:
        """Test warning when fmax exceeds Nyquist frequency."""
        mel_spec_db = np.random.randn(100, 80)

        with pytest.warns(UserWarning, match="exceeds Nyquist frequency"):
            MelSpectrogramData(
                mel_spec_db=mel_spec_db,
                n_mels=80,
                fmin=0.0,
                fmax=sample_rate,  # Exceeds Nyquist
                sample_rate=sample_rate,
                stft_params={},
                metadata={},
            )


class TestFeatureData:
    """Tests for FeatureData class."""

    def test_valid_feature_data(self) -> None:
        """Test valid feature data."""
        features = np.random.randn(100, 20)
        shape = (100, 20)

        feature = FeatureData(
            features=features,
            feature_type="mfcc",
            shape=shape,
            params={"n_mfcc": 20},
            metadata={},
        )

        assert feature.feature_type == "mfcc"
        assert feature.shape == shape

    def test_shape_mismatch(self) -> None:
        """Test shape mismatch."""
        features = np.random.randn(100, 20)

        with pytest.raises(ValueError, match="Shape mismatch"):
            FeatureData(
                features=features,
                feature_type="mfcc",
                shape=(100, 13),  # Wrong shape
                params={},
                metadata={},
            )

    def test_invalid_features_type(self) -> None:
        """Test invalid features type."""
        with pytest.raises(TypeError, match="features must be np.ndarray or torch.Tensor"):
            FeatureData(
                features=[1, 2, 3],  # type: ignore
                feature_type="mfcc",
                shape=(3,),
                params={},
                metadata={},
            )

    def test_empty_feature_type(self) -> None:
        """Test empty feature type."""
        features = np.random.randn(100, 20)

        with pytest.raises(ValueError, match="feature_type cannot be empty"):
            FeatureData(
                features=features,
                feature_type="",
                shape=(100, 20),
                params={},
                metadata={},
            )


class TestProcessingState:
    """Tests for ProcessingState class."""

    def test_valid_processing_state(self, sample_rate: int) -> None:
        """Test valid processing state."""
        waveform = np.random.randn(16000)
        audio = AudioData(
            waveform=waveform,
            sample_rate=sample_rate,
            n_channels=1,
            duration=1.0,
            metadata={},
        )

        state = ProcessingState(
            stage="loaded",
            data=audio,
            timestamp=time.time(),
            device="cpu",
        )

        assert state.stage == "loaded"
        assert state.device == "cpu"

    def test_invalid_stage(self, sample_rate: int) -> None:
        """Test invalid stage."""
        waveform = np.random.randn(16000)
        audio = AudioData(
            waveform=waveform,
            sample_rate=sample_rate,
            n_channels=1,
            duration=1.0,
            metadata={},
        )

        with pytest.raises(ValueError, match="stage must be one of"):
            ProcessingState(
                stage="invalid",
                data=audio,
                timestamp=time.time(),
                device="cpu",
            )

    def test_invalid_device(self, sample_rate: int) -> None:
        """Test invalid device."""
        waveform = np.random.randn(16000)
        audio = AudioData(
            waveform=waveform,
            sample_rate=sample_rate,
            n_channels=1,
            duration=1.0,
            metadata={},
        )

        with pytest.raises(ValueError, match="device must be 'cpu' or 'cuda"):
            ProcessingState(
                stage="loaded",
                data=audio,
                timestamp=time.time(),
                device="gpu",
            )

    def test_invalid_timestamp(self, sample_rate: int) -> None:
        """Test invalid timestamp."""
        waveform = np.random.randn(16000)
        audio = AudioData(
            waveform=waveform,
            sample_rate=sample_rate,
            n_channels=1,
            duration=1.0,
            metadata={},
        )

        with pytest.raises(ValueError, match="timestamp must be non-negative"):
            ProcessingState(
                stage="loaded",
                data=audio,
                timestamp=-1.0,
                device="cpu",
            )

    def test_describe(self, sample_rate: int) -> None:
        """Test describe method."""
        waveform = np.random.randn(16000)
        audio = AudioData(
            waveform=waveform,
            sample_rate=sample_rate,
            n_channels=1,
            duration=1.0,
            metadata={},
        )

        state = ProcessingState(
            stage="loaded",
            data=audio,
            timestamp=time.time(),
            device="cpu",
        )

        description = state.describe()
        assert "loaded" in description
        assert "AudioData" in description
        assert "cpu" in description
