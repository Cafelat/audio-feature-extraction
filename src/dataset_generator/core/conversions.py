"""NumPy ↔ PyTorch tensor conversion utilities."""

from typing import Any

import numpy as np
import torch

from dataset_generator.core.models import (
    AudioData,
    FeatureData,
    MelSpectrogramData,
    SpectrogramData,
)


class TensorConverter:
    """NumPy ↔ PyTorch conversion utility.

    This class provides static methods for converting between NumPy arrays
    and PyTorch tensors, with support for device transfer and data class
    conversion.

    Conversion Policy:
        - Internal processing: PyTorch Tensor
        - I/O boundary: NumPy ndarray
        - Data classes: Accept both NumPy and PyTorch
        - Conversion is explicit, not automatic
    """

    @staticmethod
    def to_torch(
        array: np.ndarray | torch.Tensor,
        device: str = "cpu",
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Convert NumPy array to PyTorch tensor.

        Args:
            array: Input array (NumPy or PyTorch)
            device: Target device ('cpu', 'cuda', 'cuda:0', etc.)
            dtype: Target data type (None to preserve original type)

        Returns:
            PyTorch tensor

        Raises:
            TypeError: If input is not NumPy array or PyTorch tensor
        """
        if isinstance(array, torch.Tensor):
            tensor = array
        elif isinstance(array, np.ndarray):
            # NumPy → PyTorch
            tensor = torch.from_numpy(array)
        else:
            raise TypeError(
                f"Expected np.ndarray or torch.Tensor, got {type(array)}"
            )

        # Device transfer
        tensor = tensor.to(device)

        # Data type conversion (if specified)
        if dtype is not None:
            tensor = tensor.to(dtype)

        return tensor

    @staticmethod
    def to_numpy(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
        """Convert PyTorch tensor to NumPy array.

        Args:
            tensor: Input tensor (PyTorch or NumPy)

        Returns:
            NumPy array

        Raises:
            TypeError: If input is not PyTorch tensor or NumPy array
        """
        if isinstance(tensor, np.ndarray):
            return tensor
        elif isinstance(tensor, torch.Tensor):
            # Transfer to CPU if on GPU
            if tensor.is_cuda:
                tensor = tensor.cpu()

            # Detach from computation graph
            if tensor.requires_grad:
                tensor = tensor.detach()

            # Convert to NumPy
            return tensor.numpy()
        else:
            raise TypeError(
                f"Expected torch.Tensor or np.ndarray, got {type(tensor)}"
            )

    @staticmethod
    def ensure_torch(data: Any, device: str = "cpu") -> Any:
        """Convert all arrays in data class to PyTorch tensors.

        This method converts AudioData, SpectrogramData, MelSpectrogramData,
        and FeatureData objects to use PyTorch tensors internally.

        Args:
            data: Input data class instance
            device: Target device for tensors

        Returns:
            Data class with PyTorch tensors
        """
        if isinstance(data, AudioData):
            return AudioData(
                waveform=TensorConverter.to_torch(data.waveform, device),
                sample_rate=data.sample_rate,
                n_channels=data.n_channels,
                duration=data.duration,
                metadata=data.metadata,
            )
        elif isinstance(data, SpectrogramData):
            return SpectrogramData(
                complex_spec=TensorConverter.to_torch(data.complex_spec, device),
                magnitude_db=TensorConverter.to_torch(data.magnitude_db, device),
                phase=TensorConverter.to_torch(data.phase, device),
                n_fft=data.n_fft,
                hop_length=data.hop_length,
                win_length=data.win_length,
                window=data.window,
                sample_rate=data.sample_rate,
                metadata=data.metadata,
            )
        elif isinstance(data, MelSpectrogramData):
            return MelSpectrogramData(
                mel_spec_db=TensorConverter.to_torch(data.mel_spec_db, device),
                n_mels=data.n_mels,
                fmin=data.fmin,
                fmax=data.fmax,
                sample_rate=data.sample_rate,
                stft_params=data.stft_params,
                metadata=data.metadata,
            )
        elif isinstance(data, FeatureData):
            return FeatureData(
                features=TensorConverter.to_torch(data.features, device),
                feature_type=data.feature_type,
                shape=data.shape,
                params=data.params,
                metadata=data.metadata,
            )
        else:
            return data

    @staticmethod
    def ensure_numpy(data: Any) -> Any:
        """Convert all tensors in data class to NumPy arrays.

        This method converts AudioData, SpectrogramData, MelSpectrogramData,
        and FeatureData objects to use NumPy arrays internally.

        Args:
            data: Input data class instance

        Returns:
            Data class with NumPy arrays
        """
        if isinstance(data, AudioData):
            return AudioData(
                waveform=TensorConverter.to_numpy(data.waveform),
                sample_rate=data.sample_rate,
                n_channels=data.n_channels,
                duration=data.duration,
                metadata=data.metadata,
            )
        elif isinstance(data, SpectrogramData):
            return SpectrogramData(
                complex_spec=TensorConverter.to_numpy(data.complex_spec),
                magnitude_db=TensorConverter.to_numpy(data.magnitude_db),
                phase=TensorConverter.to_numpy(data.phase),
                n_fft=data.n_fft,
                hop_length=data.hop_length,
                win_length=data.win_length,
                window=data.window,
                sample_rate=data.sample_rate,
                metadata=data.metadata,
            )
        elif isinstance(data, MelSpectrogramData):
            return MelSpectrogramData(
                mel_spec_db=TensorConverter.to_numpy(data.mel_spec_db),
                n_mels=data.n_mels,
                fmin=data.fmin,
                fmax=data.fmax,
                sample_rate=data.sample_rate,
                stft_params=data.stft_params,
                metadata=data.metadata,
            )
        elif isinstance(data, FeatureData):
            return FeatureData(
                features=TensorConverter.to_numpy(data.features),
                feature_type=data.feature_type,
                shape=data.shape,
                params=data.params,
                metadata=data.metadata,
            )
        else:
            return data
