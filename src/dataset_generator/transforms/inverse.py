"""Inverse transforms for audio reconstruction."""

import numpy as np
import torch

from ..core.models import AudioData, SpectrogramData


class ISTFTReconstructor:
    """ISTFT-based audio reconstructor.

    Reconstructs audio waveform from spectrogram using inverse STFT.
    Supports both complex spectrogram and magnitude+phase inputs.
    """

    def __init__(self, device: str = "cpu"):
        """Initialize ISTFT reconstructor.

        Args:
            device: Device for computation ('cpu' or 'cuda')
        """
        self.device = device

    def reconstruct(self, spec: SpectrogramData) -> AudioData:
        """Reconstruct audio from spectrogram.

        Args:
            spec: Spectrogram data (with magnitude_db + phase or complex_spec)

        Returns:
            Reconstructed audio data

        Notes:
            If complex_spec is available, it's used directly for ISTFT.
            Otherwise, reconstructs complex spectrogram from magnitude_db and phase.
        """
        # Use complex spectrogram if available
        if spec.complex_spec is not None:
            complex_spec = self._to_device(spec.complex_spec)
        else:
            # Reconstruct from magnitude_db and phase
            if spec.magnitude_db is None or spec.phase is None:
                raise ValueError(
                    "Either complex_spec or both magnitude_db and phase must be provided"
                )

            magnitude_db = self._to_device(spec.magnitude_db)
            phase = self._to_device(spec.phase)

            # Convert dB to linear scale
            magnitude = self._from_db(magnitude_db)

            # Reconstruct complex spectrogram
            complex_spec = magnitude * torch.exp(1j * phase)

        # Get window function
        window = self._get_window(spec.window, spec.win_length)

        # torch.istft expects (freq, time) or (batch, freq, time)
        # Our spec is (time, freq), so transpose
        if complex_spec.ndim == 2:
            complex_spec = complex_spec.T  # (time, freq) -> (freq, time)
        elif complex_spec.ndim == 3:
            complex_spec = complex_spec.permute(0, 2, 1)  # (batch, time, freq) -> (batch, freq, time)

        # Perform ISTFT
        waveform = torch.istft(
            complex_spec,
            n_fft=spec.n_fft,
            hop_length=spec.hop_length,
            win_length=spec.win_length,
            window=window,
            center=True,
        )

        # Normalize to prevent clipping
        waveform = self._normalize(waveform)

        # Calculate duration
        duration = len(waveform) / spec.sample_rate

        return AudioData(
            waveform=waveform.cpu().numpy(),
            sample_rate=spec.sample_rate,
            n_channels=1,
            duration=duration,
            metadata={
                "reconstructed": True,
                "method": "istft",
                "n_fft": spec.n_fft,
                "hop_length": spec.hop_length,
            },
        )

    def _from_db(
        self, magnitude_db: torch.Tensor, ref: float = 1.0
    ) -> torch.Tensor:
        """Convert dB scale to linear scale.

        Args:
            magnitude_db: Magnitude in dB scale
            ref: Reference value used in dB conversion

        Returns:
            Magnitude in linear scale
        """
        return ref * torch.pow(10.0, magnitude_db / 20.0)

    def _get_window(self, window_type: str, win_length: int) -> torch.Tensor:
        """Get window function.

        Args:
            window_type: Window type ('hann', 'hamming', 'blackman')
            win_length: Window length

        Returns:
            Window tensor
        """
        window_type = window_type.lower()

        if window_type == "hann":
            window = torch.hann_window(win_length, device=self.device)
        elif window_type == "hamming":
            window = torch.hamming_window(win_length, device=self.device)
        elif window_type == "blackman":
            window = torch.blackman_window(win_length, device=self.device)
        else:
            # Default to Hann window
            window = torch.hann_window(win_length, device=self.device)

        return window

    def _normalize(self, waveform: torch.Tensor, headroom: float = 0.1) -> torch.Tensor:
        """Normalize waveform to prevent clipping.

        Args:
            waveform: Input waveform
            headroom: Headroom to prevent clipping (0.1 = 10% margin)

        Returns:
            Normalized waveform
        """
        max_val = torch.abs(waveform).max()
        if max_val > 1.0 - headroom:
            waveform = waveform / max_val * (1.0 - headroom)
        return waveform

    def _to_device(self, tensor: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Transfer tensor to target device.

        Args:
            tensor: NumPy array or PyTorch tensor

        Returns:
            PyTorch tensor on target device
        """
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        return tensor.to(self.device)


class GriffinLimReconstructor:
    """Griffin-Lim algorithm for phase reconstruction.
    
    Reconstructs audio from magnitude spectrogram only, estimating phase
    through iterative STFT/ISTFT cycles.
    """
    
    def __init__(self, n_iter: int = 32, momentum: float = 0.99, device: str = 'cpu'):
        """Initialize Griffin-Lim reconstructor.
        
        Args:
            n_iter: Number of iterations
            momentum: Momentum coefficient for phase update
            device: Device for computation ('cpu' or 'cuda')
        """
        self.n_iter = n_iter
        self.momentum = momentum
        self.device = device
    
    def reconstruct(self, spec: SpectrogramData) -> AudioData:
        """Reconstruct audio from magnitude spectrogram using Griffin-Lim.
        
        Args:
            spec: Spectrogram data (magnitude_db + phase, or complex_spec)
            
        Returns:
            Reconstructed audio data
        """
        # Extract magnitude from dB scale
        if spec.magnitude_db is not None:
            magnitude_db = torch.as_tensor(spec.magnitude_db, device=self.device)
            magnitude = self._from_db(magnitude_db)
        elif spec.complex_spec is not None:
            complex_spec = torch.as_tensor(spec.complex_spec, device=self.device)
            magnitude = torch.abs(complex_spec)
        else:
            raise ValueError("Either magnitude_db or complex_spec must be provided")
        
        # Ensure magnitude is (freq, time) format for torch.istft
        if magnitude.ndim == 2:
            if magnitude.shape[0] < magnitude.shape[1]:
                # Assume (time, freq) format, transpose to (freq, time)
                magnitude = magnitude.T
        
        # Initialize phase randomly
        phase = torch.rand_like(magnitude) * 2 * np.pi - np.pi
        
        # Get window function
        window = self._get_window(spec.window, spec.win_length)
        
        # Griffin-Lim iterations
        for _ in range(self.n_iter):
            # Create complex spectrogram
            complex_spec = magnitude * torch.exp(1j * phase)
            
            # ISTFT -> waveform
            waveform = torch.istft(
                complex_spec,
                n_fft=spec.n_fft,
                hop_length=spec.hop_length,
                win_length=spec.win_length,
                window=window,
                center=True
            )
            
            # STFT -> new complex spectrogram
            new_complex_spec = torch.stft(
                waveform,
                n_fft=spec.n_fft,
                hop_length=spec.hop_length,
                win_length=spec.win_length,
                window=window,
                center=True,
                return_complex=True
            )
            
            # Update phase with momentum
            new_phase = torch.angle(new_complex_spec)
            phase = self.momentum * phase + (1 - self.momentum) * new_phase
        
        # Final reconstruction
        complex_spec = magnitude * torch.exp(1j * phase)
        waveform = torch.istft(
            complex_spec,
            n_fft=spec.n_fft,
            hop_length=spec.hop_length,
            win_length=spec.win_length,
            window=window,
            center=True
        )
        
        # Normalize
        max_val = torch.abs(waveform).max()
        if max_val > 1.0:
            waveform = waveform / max_val
        
        # Calculate duration
        duration = len(waveform) / spec.sample_rate
        
        return AudioData(
            waveform=waveform.cpu().numpy(),
            sample_rate=spec.sample_rate,
            n_channels=1,
            duration=duration,
            metadata={
                'reconstructed': True,
                'method': 'griffin-lim',
                'n_iter': self.n_iter
            }
        )
    
    def _from_db(self, magnitude_db: torch.Tensor, ref: float = 1.0) -> torch.Tensor:
        """Convert dB scale to linear scale."""
        return ref * torch.pow(10.0, magnitude_db / 20.0)
    
    def _get_window(self, window_type: str, win_length: int) -> torch.Tensor:
        """Get window function."""
        window_type = window_type.lower()
        
        if window_type == "hann":
            window = torch.hann_window(win_length, device=self.device)
        elif window_type == "hamming":
            window = torch.hamming_window(win_length, device=self.device)
        elif window_type == "blackman":
            window = torch.blackman_window(win_length, device=self.device)
        else:
            window = torch.hann_window(win_length, device=self.device)
        
        return window
