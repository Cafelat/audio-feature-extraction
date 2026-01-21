"""STFT feature extraction module."""

import torch

from dataset_generator.core.conversions import TensorConverter
from dataset_generator.core.models import AudioData, SpectrogramData


class STFTExtractor:
    """STFT feature extractor.

    Extracts magnitude and phase spectrograms with dB scale conversion.
    Supports GPU acceleration.

    Attributes:
        n_fft: FFT size
        hop_length: Hop length in samples
        win_length: Window length in samples
        window: Window function name
        center: Enable center padding
        device: Target device ('cpu' or 'cuda')
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: int | None = None,
        window: str = "hann",
        center: bool = True,
        device: str = "cpu",
    ):
        """Initialize STFT extractor.

        Args:
            n_fft: FFT size
            hop_length: Hop length in samples
            win_length: Window length (None to use n_fft)
            window: Window function ('hann', 'hamming', 'blackman', etc.)
            center: Enable center padding
            device: Target device ('cpu', 'cuda', 'cuda:0', etc.)
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.window = window
        self.center = center
        self.device = device

    def extract(self, audio: AudioData) -> SpectrogramData:
        """Extract STFT features.

        Args:
            audio: Input audio data

        Returns:
            SpectrogramData containing complex spec, magnitude (dB), and phase
        """
        # Transfer to device
        waveform = self._to_device(audio.waveform)

        # Handle stereo by processing first channel
        if waveform.ndim == 2:
            waveform = waveform[0]  # Take first channel

        # Compute STFT
        complex_spec = self._compute_stft(waveform)

        # Extract magnitude and phase
        magnitude = torch.abs(complex_spec)
        phase = torch.angle(complex_spec)

        # Convert to dB scale
        magnitude_db = self._to_db(magnitude)

        return SpectrogramData(
            complex_spec=complex_spec,
            magnitude_db=magnitude_db,
            phase=phase,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            sample_rate=audio.sample_rate,
            metadata={
                "source_file": audio.metadata.get("file_path"),
                "device": self.device,
            },
        )

    def get_params(self) -> dict[str, int | str | bool]:
        """Get extraction parameters.

        Returns:
            Dictionary of STFT parameters
        """
        return {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "window": self.window,
            "center": self.center,
        }

    def _compute_stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute STFT using torch.stft.

        Args:
            waveform: Input waveform (1D tensor)

        Returns:
            Complex spectrogram (n_frames, n_freq_bins)
        """
        # Create window
        if self.window == "hann":
            window_tensor = torch.hann_window(self.win_length, device=self.device)
        elif self.window == "hamming":
            window_tensor = torch.hamming_window(self.win_length, device=self.device)
        elif self.window == "blackman":
            window_tensor = torch.blackman_window(self.win_length, device=self.device)
        else:
            # Default to Hann window
            window_tensor = torch.hann_window(self.win_length, device=self.device)

        # Compute STFT
        stft_result = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window_tensor,
            center=self.center,
            return_complex=True,
        )

        # Transpose to (n_frames, n_freq_bins)
        stft_result = stft_result.T

        return stft_result

    def _to_db(
        self,
        magnitude: torch.Tensor,
        ref: float = 1.0,
        amin: float = 1e-10,
        top_db: float = 80.0,
    ) -> torch.Tensor:
        """Convert magnitude to dB scale.

        Args:
            magnitude: Magnitude spectrogram
            ref: Reference value for dB calculation
            amin: Minimum amplitude to avoid log(0)
            top_db: Maximum dB range (values below max - top_db are clamped)

        Returns:
            Magnitude in dB scale
        """
        # Clamp to avoid log(0)
        magnitude = torch.clamp(magnitude, min=amin)

        # Convert to dB: 20 * log10(magnitude / ref)
        db = 20.0 * torch.log10(magnitude / ref)

        # Clamp to top_db range
        if top_db is not None:
            db = torch.clamp(db, min=db.max() - top_db)

        return db

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transfer tensor to device.

        Args:
            tensor: Input tensor (can be NumPy array or PyTorch tensor)

        Returns:
            Tensor on target device
        """
        # Use TensorConverter if it's a NumPy array
        if not isinstance(tensor, torch.Tensor):
            tensor = TensorConverter.to_torch(tensor, device=self.device)
        else:
            tensor = tensor.to(self.device)

        return tensor
