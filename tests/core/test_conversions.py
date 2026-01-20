"""Tests for TensorConverter."""

import numpy as np
import pytest
import torch

from dataset_generator.core.conversions import TensorConverter
from dataset_generator.core.models import (
    AudioData,
    FeatureData,
    MelSpectrogramData,
    SpectrogramData,
)


class TestToTorch:
    """Tests for to_torch method."""

    def test_numpy_to_torch_cpu(self) -> None:
        """Test NumPy to PyTorch conversion on CPU."""
        array = np.random.randn(100, 50).astype(np.float32)
        tensor = TensorConverter.to_torch(array, device="cpu")

        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == "cpu"
        assert tensor.dtype == torch.float32
        assert tensor.shape == (100, 50)
        np.testing.assert_allclose(tensor.numpy(), array)

    def test_torch_to_torch(self) -> None:
        """Test PyTorch to PyTorch (identity)."""
        original = torch.randn(100, 50)
        tensor = TensorConverter.to_torch(original, device="cpu")

        assert isinstance(tensor, torch.Tensor)
        assert torch.allclose(tensor, original)

    def test_complex_numpy_to_torch(self) -> None:
        """Test complex NumPy array conversion."""
        array = np.random.randn(100, 50) + 1j * np.random.randn(100, 50)
        array = array.astype(np.complex64)
        tensor = TensorConverter.to_torch(array, device="cpu")

        assert isinstance(tensor, torch.Tensor)
        assert tensor.is_complex()
        assert tensor.dtype == torch.complex64

    def test_dtype_conversion(self) -> None:
        """Test data type conversion."""
        array = np.random.randn(100, 50).astype(np.float32)
        tensor = TensorConverter.to_torch(array, device="cpu", dtype=torch.float64)

        assert tensor.dtype == torch.float64

    def test_invalid_input_type(self) -> None:
        """Test invalid input type raises TypeError."""
        with pytest.raises(TypeError, match="Expected np.ndarray or torch.Tensor"):
            TensorConverter.to_torch([1, 2, 3], device="cpu")  # type: ignore

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_numpy_to_torch_cuda(self) -> None:
        """Test NumPy to PyTorch conversion on CUDA."""
        array = np.random.randn(100, 50).astype(np.float32)
        tensor = TensorConverter.to_torch(array, device="cuda:0")

        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == "cuda"
        assert tensor.dtype == torch.float32


class TestToNumpy:
    """Tests for to_numpy method."""

    def test_torch_to_numpy(self) -> None:
        """Test PyTorch to NumPy conversion."""
        tensor = torch.randn(100, 50)
        array = TensorConverter.to_numpy(tensor)

        assert isinstance(array, np.ndarray)
        assert array.shape == (100, 50)
        np.testing.assert_allclose(array, tensor.numpy())

    def test_numpy_to_numpy(self) -> None:
        """Test NumPy to NumPy (identity)."""
        original = np.random.randn(100, 50)
        array = TensorConverter.to_numpy(original)

        assert isinstance(array, np.ndarray)
        assert array is original  # Should be the same object

    def test_complex_torch_to_numpy(self) -> None:
        """Test complex PyTorch tensor conversion."""
        tensor = torch.randn(100, 50, dtype=torch.complex64)
        array = TensorConverter.to_numpy(tensor)

        assert isinstance(array, np.ndarray)
        assert np.iscomplexobj(array)
        assert array.dtype == np.complex64

    def test_gradient_detachment(self) -> None:
        """Test that gradient is detached during conversion."""
        tensor = torch.randn(100, 50, requires_grad=True)
        array = TensorConverter.to_numpy(tensor)

        assert isinstance(array, np.ndarray)
        # Verify conversion succeeds (detachment worked)
        assert array.shape == (100, 50)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_to_numpy(self) -> None:
        """Test CUDA tensor to NumPy conversion."""
        tensor = torch.randn(100, 50, device="cuda:0")
        array = TensorConverter.to_numpy(tensor)

        assert isinstance(array, np.ndarray)
        assert array.shape == (100, 50)

    def test_invalid_input_type(self) -> None:
        """Test invalid input type raises TypeError."""
        with pytest.raises(TypeError, match="Expected torch.Tensor or np.ndarray"):
            TensorConverter.to_numpy([1, 2, 3])  # type: ignore


class TestEnsureTorch:
    """Tests for ensure_torch method."""

    def test_audio_data_numpy_to_torch(self, sample_rate: int) -> None:
        """Test AudioData conversion from NumPy to PyTorch."""
        waveform = np.random.randn(16000).astype(np.float32)
        audio_numpy = AudioData(
            waveform=waveform,
            sample_rate=sample_rate,
            n_channels=1,
            duration=1.0,
            metadata={},
        )

        audio_torch = TensorConverter.ensure_torch(audio_numpy, device="cpu")

        assert isinstance(audio_torch.waveform, torch.Tensor)
        assert audio_torch.waveform.device.type == "cpu"
        assert audio_torch.sample_rate == sample_rate

    def test_audio_data_torch_to_torch(self, sample_rate: int) -> None:
        """Test AudioData with PyTorch tensor remains PyTorch."""
        waveform = torch.randn(16000)
        audio_torch = AudioData(
            waveform=waveform,
            sample_rate=sample_rate,
            n_channels=1,
            duration=1.0,
            metadata={},
        )

        audio_result = TensorConverter.ensure_torch(audio_torch, device="cpu")

        assert isinstance(audio_result.waveform, torch.Tensor)

    def test_spectrogram_data_conversion(self, sample_rate: int) -> None:
        """Test SpectrogramData conversion."""
        n_fft = 512
        n_freq_bins = n_fft // 2 + 1

        complex_spec = np.random.randn(100, n_freq_bins) + 1j * np.random.randn(
            100, n_freq_bins
        )
        magnitude_db = np.random.randn(100, n_freq_bins).astype(np.float32)
        phase = np.random.randn(100, n_freq_bins).astype(np.float32)

        spec_numpy = SpectrogramData(
            complex_spec=complex_spec.astype(np.complex64),
            magnitude_db=magnitude_db,
            phase=phase,
            n_fft=n_fft,
            hop_length=256,
            win_length=512,
            window="hann",
            sample_rate=sample_rate,
            metadata={},
        )

        spec_torch = TensorConverter.ensure_torch(spec_numpy, device="cpu")

        assert isinstance(spec_torch.complex_spec, torch.Tensor)
        assert isinstance(spec_torch.magnitude_db, torch.Tensor)
        assert isinstance(spec_torch.phase, torch.Tensor)
        assert spec_torch.complex_spec.is_complex()

    def test_mel_spectrogram_data_conversion(self, sample_rate: int) -> None:
        """Test MelSpectrogramData conversion."""
        mel_spec_db = np.random.randn(100, 80).astype(np.float32)

        mel_numpy = MelSpectrogramData(
            mel_spec_db=mel_spec_db,
            n_mels=80,
            fmin=0.0,
            fmax=8000.0,
            sample_rate=sample_rate,
            stft_params={"n_fft": 512},
            metadata={},
        )

        mel_torch = TensorConverter.ensure_torch(mel_numpy, device="cpu")

        assert isinstance(mel_torch.mel_spec_db, torch.Tensor)
        assert mel_torch.n_mels == 80

    def test_feature_data_conversion(self) -> None:
        """Test FeatureData conversion."""
        features = np.random.randn(100, 20).astype(np.float32)

        feature_numpy = FeatureData(
            features=features,
            feature_type="mfcc",
            shape=(100, 20),
            params={},
            metadata={},
        )

        feature_torch = TensorConverter.ensure_torch(feature_numpy, device="cpu")

        assert isinstance(feature_torch.features, torch.Tensor)

    def test_unsupported_type_passthrough(self) -> None:
        """Test unsupported type is passed through."""
        data = {"key": "value"}
        result = TensorConverter.ensure_torch(data, device="cpu")
        assert result is data

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_ensure_torch_cuda(self, sample_rate: int) -> None:
        """Test ensure_torch with CUDA device."""
        waveform = np.random.randn(16000).astype(np.float32)
        audio_numpy = AudioData(
            waveform=waveform,
            sample_rate=sample_rate,
            n_channels=1,
            duration=1.0,
            metadata={},
        )

        audio_cuda = TensorConverter.ensure_torch(audio_numpy, device="cuda:0")

        assert isinstance(audio_cuda.waveform, torch.Tensor)
        assert audio_cuda.waveform.device.type == "cuda"


class TestEnsureNumpy:
    """Tests for ensure_numpy method."""

    def test_audio_data_torch_to_numpy(self, sample_rate: int) -> None:
        """Test AudioData conversion from PyTorch to NumPy."""
        waveform = torch.randn(16000)
        audio_torch = AudioData(
            waveform=waveform,
            sample_rate=sample_rate,
            n_channels=1,
            duration=1.0,
            metadata={},
        )

        audio_numpy = TensorConverter.ensure_numpy(audio_torch)

        assert isinstance(audio_numpy.waveform, np.ndarray)
        assert audio_numpy.sample_rate == sample_rate

    def test_audio_data_numpy_to_numpy(self, sample_rate: int) -> None:
        """Test AudioData with NumPy array remains NumPy."""
        waveform = np.random.randn(16000).astype(np.float32)
        audio_numpy = AudioData(
            waveform=waveform,
            sample_rate=sample_rate,
            n_channels=1,
            duration=1.0,
            metadata={},
        )

        audio_result = TensorConverter.ensure_numpy(audio_numpy)

        assert isinstance(audio_result.waveform, np.ndarray)

    def test_spectrogram_data_conversion(self, sample_rate: int) -> None:
        """Test SpectrogramData conversion."""
        n_fft = 512
        n_freq_bins = n_fft // 2 + 1

        complex_spec = torch.randn(100, n_freq_bins, dtype=torch.complex64)
        magnitude_db = torch.randn(100, n_freq_bins)
        phase = torch.randn(100, n_freq_bins)

        spec_torch = SpectrogramData(
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

        spec_numpy = TensorConverter.ensure_numpy(spec_torch)

        assert isinstance(spec_numpy.complex_spec, np.ndarray)
        assert isinstance(spec_numpy.magnitude_db, np.ndarray)
        assert isinstance(spec_numpy.phase, np.ndarray)
        assert np.iscomplexobj(spec_numpy.complex_spec)

    def test_mel_spectrogram_data_conversion(self, sample_rate: int) -> None:
        """Test MelSpectrogramData conversion."""
        mel_spec_db = torch.randn(100, 80)

        mel_torch = MelSpectrogramData(
            mel_spec_db=mel_spec_db,
            n_mels=80,
            fmin=0.0,
            fmax=8000.0,
            sample_rate=sample_rate,
            stft_params={"n_fft": 512},
            metadata={},
        )

        mel_numpy = TensorConverter.ensure_numpy(mel_torch)

        assert isinstance(mel_numpy.mel_spec_db, np.ndarray)

    def test_feature_data_conversion(self) -> None:
        """Test FeatureData conversion."""
        features = torch.randn(100, 20)

        feature_torch = FeatureData(
            features=features,
            feature_type="mfcc",
            shape=(100, 20),
            params={},
            metadata={},
        )

        feature_numpy = TensorConverter.ensure_numpy(feature_torch)

        assert isinstance(feature_numpy.features, np.ndarray)

    def test_unsupported_type_passthrough(self) -> None:
        """Test unsupported type is passed through."""
        data = {"key": "value"}
        result = TensorConverter.ensure_numpy(data)
        assert result is data

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_ensure_numpy_from_cuda(self, sample_rate: int) -> None:
        """Test ensure_numpy from CUDA tensor."""
        waveform = torch.randn(16000, device="cuda:0")
        audio_cuda = AudioData(
            waveform=waveform,
            sample_rate=sample_rate,
            n_channels=1,
            duration=1.0,
            metadata={},
        )

        audio_numpy = TensorConverter.ensure_numpy(audio_cuda)

        assert isinstance(audio_numpy.waveform, np.ndarray)


class TestRoundTrip:
    """Tests for round-trip conversion."""

    def test_numpy_torch_numpy_roundtrip(self) -> None:
        """Test NumPy → PyTorch → NumPy round trip."""
        original = np.random.randn(100, 50).astype(np.float32)
        tensor = TensorConverter.to_torch(original, device="cpu")
        result = TensorConverter.to_numpy(tensor)

        np.testing.assert_allclose(result, original)

    def test_torch_numpy_torch_roundtrip(self) -> None:
        """Test PyTorch → NumPy → PyTorch round trip."""
        original = torch.randn(100, 50)
        array = TensorConverter.to_numpy(original)
        result = TensorConverter.to_torch(array, device="cpu")

        torch.testing.assert_close(result, original)

    def test_audio_data_roundtrip(self, sample_rate: int) -> None:
        """Test AudioData round-trip conversion."""
        waveform = np.random.randn(16000).astype(np.float32)
        audio_original = AudioData(
            waveform=waveform,
            sample_rate=sample_rate,
            n_channels=1,
            duration=1.0,
            metadata={"test": "value"},
        )

        audio_torch = TensorConverter.ensure_torch(audio_original, device="cpu")
        audio_result = TensorConverter.ensure_numpy(audio_torch)

        assert isinstance(audio_result.waveform, np.ndarray)
        np.testing.assert_allclose(audio_result.waveform, waveform)
        assert audio_result.metadata == audio_original.metadata
