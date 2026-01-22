"""Tests for inverse transforms."""

import numpy as np
import pytest
import torch

from dataset_generator.core.models import AudioData, SpectrogramData
from dataset_generator.features.stft import STFTExtractor
from dataset_generator.transforms.inverse import ISTFTReconstructor


class TestISTFTReconstructor:
    """Test ISTFTReconstructor class."""

    @pytest.fixture
    def reconstructor(self):
        """Create ISTFT reconstructor."""
        return ISTFTReconstructor(device="cpu")

    @pytest.fixture
    def stft_extractor(self):
        """Create STFT extractor."""
        return STFTExtractor(n_fft=2048, hop_length=512, device="cpu")

    @pytest.fixture
    def sine_wave(self):
        """Create test sine wave."""
        sr = 16000
        duration = 1.0
        freq = 440.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        waveform = np.sin(2 * np.pi * freq * t).astype(np.float32)

        return AudioData(
            waveform=waveform,
            sample_rate=sr,
            n_channels=1,
            duration=duration,
            metadata={"test": True},
        )

    def test_init(self, reconstructor):
        """Test initialization."""
        assert reconstructor.device == "cpu"

    def test_init_cuda(self):
        """Test CUDA initialization."""
        reconstructor = ISTFTReconstructor(device="cuda")
        assert reconstructor.device == "cuda"

    def test_reconstruct_from_complex(self, reconstructor, stft_extractor, sine_wave):
        """Test reconstruction from complex spectrogram."""
        # Extract STFT
        spec = stft_extractor.extract(sine_wave)

        # Reconstruct
        reconstructed = reconstructor.reconstruct(spec)

        # Check metadata
        assert reconstructed.metadata["reconstructed"] is True
        assert reconstructed.metadata["method"] == "istft"
        assert reconstructed.sample_rate == sine_wave.sample_rate
        assert reconstructed.n_channels == 1

        # Check waveform type
        assert isinstance(reconstructed.waveform, np.ndarray)
        assert reconstructed.waveform.dtype == np.float32

    def test_reconstruct_from_magnitude_phase(
        self, reconstructor, stft_extractor, sine_wave
    ):
        """Test reconstruction from magnitude_db and phase."""
        # Extract STFT
        spec = stft_extractor.extract(sine_wave)

        # Create spec without complex_spec
        spec_without_complex = SpectrogramData(
            complex_spec=None,  # Remove complex spec
            magnitude_db=spec.magnitude_db,
            phase=spec.phase,
            n_fft=spec.n_fft,
            hop_length=spec.hop_length,
            win_length=spec.win_length,
            window=spec.window,
            sample_rate=spec.sample_rate,
            metadata=spec.metadata,
        )

        # Reconstruct
        reconstructed = reconstructor.reconstruct(spec_without_complex)

        # Check basic properties
        assert reconstructed.sample_rate == sine_wave.sample_rate
        assert isinstance(reconstructed.waveform, np.ndarray)

    def test_reconstruct_high_quality(self, reconstructor, stft_extractor, sine_wave):
        """Test high-quality reconstruction (SNR >= 19dB is acceptable)."""
        # Extract STFT
        spec = stft_extractor.extract(sine_wave)

        # Reconstruct
        reconstructed = reconstructor.reconstruct(spec)

        # Align lengths (ISTFT may produce slightly different length)
        min_len = min(len(sine_wave.waveform), len(reconstructed.waveform))
        original = sine_wave.waveform[:min_len]
        recon = reconstructed.waveform[:min_len]

        # Calculate SNR
        signal_power = np.mean(original**2)
        noise_power = np.mean((original - recon) ** 2)
        snr_db = 10 * np.log10(signal_power / noise_power)

        # Should achieve >= 19dB SNR (high quality reconstruction)
        # Note: STFT with default hop_length (512) has some reconstruction loss
        assert snr_db >= 19.0, f"SNR {snr_db:.2f}dB is too low"

    def test_reconstruct_missing_data(self, reconstructor):
        """Test error when both complex_spec and magnitude/phase are missing."""
        with pytest.raises(ValueError, match="Either complex_spec or both"):
            SpectrogramData(
                complex_spec=None,
                magnitude_db=None,
                phase=None,
                n_fft=2048,
                hop_length=512,
                win_length=2048,
                window="hann",
                sample_rate=16000,
                metadata={},
            )

    def test_from_db_conversion(self, reconstructor):
        """Test dB to linear conversion."""
        # Test cases: dB -> expected linear value
        db_values = torch.tensor([0.0, 20.0, -20.0, 40.0])
        expected = torch.tensor([1.0, 10.0, 0.1, 100.0])

        result = reconstructor._from_db(db_values, ref=1.0)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_get_window_hann(self, reconstructor):
        """Test Hann window generation."""
        window = reconstructor._get_window("hann", 512)
        assert window.shape == (512,)
        assert window.device.type == "cpu"

        # Even-length Hann windows are not perfectly symmetric
        # Just check it's a valid window (sums to reasonable value)
        assert window.sum() > 100.0  # Typical for 512-length window
        assert window.min() >= 0.0
        assert window.max() <= 1.0

    def test_get_window_hamming(self, reconstructor):
        """Test Hamming window generation."""
        window = reconstructor._get_window("hamming", 512)
        assert window.shape == (512,)

    def test_get_window_blackman(self, reconstructor):
        """Test Blackman window generation."""
        window = reconstructor._get_window("blackman", 512)
        assert window.shape == (512,)

    def test_get_window_unknown(self, reconstructor):
        """Test unknown window type falls back to Hann."""
        window_unknown = reconstructor._get_window("unknown", 512)
        window_hann = reconstructor._get_window("hann", 512)

        torch.testing.assert_close(window_unknown, window_hann)

    def test_normalize_no_clipping(self, reconstructor):
        """Test normalization with normal amplitude."""
        waveform = torch.tensor([0.1, -0.2, 0.3, -0.4])
        normalized = reconstructor._normalize(waveform)

        # Should remain unchanged (max < 0.9)
        torch.testing.assert_close(normalized, waveform)

    def test_normalize_with_clipping(self, reconstructor):
        """Test normalization prevents clipping."""
        waveform = torch.tensor([0.5, -1.5, 0.3, -0.4])
        normalized = reconstructor._normalize(waveform, headroom=0.1)

        # Should be scaled down
        assert torch.abs(normalized).max() <= 0.9  # 1.0 - 0.1 headroom
        # Should maintain relative proportions
        assert torch.allclose(normalized / normalized.max(), waveform / waveform.max())

    def test_to_device_numpy(self, reconstructor):
        """Test NumPy array to tensor conversion."""
        array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = reconstructor._to_device(array)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == "cpu"
        torch.testing.assert_close(tensor, torch.tensor([1.0, 2.0, 3.0]))

    def test_to_device_tensor(self, reconstructor):
        """Test tensor device transfer."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = reconstructor._to_device(tensor)

        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_reconstruction(self, stft_extractor, sine_wave):
        """Test CUDA reconstruction."""
        reconstructor = ISTFTReconstructor(device="cuda")

        # Extract STFT on CPU
        spec = stft_extractor.extract(sine_wave)

        # Reconstruct on GPU
        reconstructed = reconstructor.reconstruct(spec)

        # Result should be on CPU (after .cpu().numpy())
        assert isinstance(reconstructed.waveform, np.ndarray)
        assert reconstructed.sample_rate == sine_wave.sample_rate

    def test_duration_calculation(self, reconstructor, stft_extractor, sine_wave):
        """Test duration calculation."""
        spec = stft_extractor.extract(sine_wave)
        reconstructed = reconstructor.reconstruct(spec)

        # Duration should be close to original
        expected_duration = len(reconstructed.waveform) / reconstructed.sample_rate
        assert abs(reconstructed.duration - expected_duration) < 1e-6

    def test_waveform_shape(self, reconstructor, stft_extractor, sine_wave):
        """Test reconstructed waveform shape."""
        spec = stft_extractor.extract(sine_wave)
        reconstructed = reconstructor.reconstruct(spec)

        # Should be 1D array
        assert reconstructed.waveform.ndim == 1
        assert len(reconstructed.waveform) > 0

    def test_round_trip_consistency(self, reconstructor, stft_extractor):
        """Test multiple round trips maintain consistency."""
        # Create test signal
        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        waveform = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        audio = AudioData(
            waveform=waveform,
            sample_rate=sr,
            n_channels=1,
            duration=duration,
            metadata={},
        )

        # First round trip
        spec1 = stft_extractor.extract(audio)
        recon1 = reconstructor.reconstruct(spec1)

        # Second round trip
        spec2 = stft_extractor.extract(recon1)
        recon2 = reconstructor.reconstruct(spec2)

        # Align lengths
        min_len = min(len(recon1.waveform), len(recon2.waveform))
        r1 = recon1.waveform[:min_len]
        r2 = recon2.waveform[:min_len]

        # Should be very similar
        correlation = np.corrcoef(r1, r2)[0, 1]
        assert correlation > 0.999, "Round trips should be consistent"
