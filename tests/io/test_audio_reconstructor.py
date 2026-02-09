"""Tests for AudioReconstructor class"""

import numpy as np
import pytest
import torch
import tempfile
import os
from pathlib import Path

from dataset_generator.io.audio_reconstructor import AudioReconstructor
from dataset_generator.core.models import AudioData, SpectrogramData
from dataset_generator.core.types import DatasetFormat
from dataset_generator.transforms.inverse import ISTFTReconstructor


@pytest.fixture
def reconstructor():
    """Create an AudioReconstructor instance"""
    return AudioReconstructor(device='cpu')


@pytest.fixture
def stft_params():
    """Standard STFT parameters"""
    return {
        'n_fft': 2048,
        'hop_length': 512,
        'win_length': 2048,
        'window': 'hann',
        'sample_rate': 22050
    }


@pytest.fixture
def sample_magnitude_phase_trig():
    """Create a sample model output in MAGNITUDE_PHASE_TRIG format"""
    # Shape: (time_frames, freq_bins, 3)
    time_frames = 100
    freq_bins = 1025  # n_fft // 2 + 1
    
    magnitude_db = np.random.uniform(-80, 0, (time_frames, freq_bins))
    cos_phase = np.random.uniform(-1, 1, (time_frames, freq_bins))
    sin_phase = np.random.uniform(-1, 1, (time_frames, freq_bins))
    
    # Normalize to unit circle
    norm = np.sqrt(cos_phase**2 + sin_phase**2) + 1e-10
    cos_phase = cos_phase / norm
    sin_phase = sin_phase / norm
    
    data = np.stack([magnitude_db, cos_phase, sin_phase], axis=-1)
    return data.astype(np.float32)


@pytest.fixture
def sample_complex():
    """Create a sample model output in COMPLEX format"""
    time_frames = 100
    freq_bins = 1025
    
    real = np.random.randn(time_frames, freq_bins).astype(np.float32)
    imag = np.random.randn(time_frames, freq_bins).astype(np.float32)
    
    data = np.stack([real, imag], axis=-1)
    return data.astype(np.float32)


@pytest.fixture
def sample_magnitude_phase():
    """Create a sample model output in MAGNITUDE_PHASE format"""
    time_frames = 100
    freq_bins = 1025
    
    magnitude_db = np.random.uniform(-80, 0, (time_frames, freq_bins))
    phase = np.random.uniform(-np.pi, np.pi, (time_frames, freq_bins))
    
    data = np.stack([magnitude_db, phase], axis=-1)
    return data.astype(np.float32)


@pytest.fixture
def sample_magnitude_only():
    """Create a sample model output in MAGNITUDE_ONLY format"""
    time_frames = 100
    freq_bins = 1025
    
    magnitude_db = np.random.uniform(-80, 0, (time_frames, freq_bins))
    
    data = magnitude_db[..., np.newaxis]
    return data.astype(np.float32)


class TestAudioReconstructorInit:
    """Test AudioReconstructor initialization"""
    
    def test_init_cpu(self):
        """Test initialization with CPU device"""
        reconstructor = AudioReconstructor(device='cpu')
        assert reconstructor.device == 'cpu'
        assert reconstructor.istft_reconstructor is not None
        assert reconstructor.griffin_lim_reconstructor is not None
        assert reconstructor.converter is not None
    
    def test_init_cuda(self):
        """Test initialization with CUDA device (will work even if CUDA not available)"""
        reconstructor = AudioReconstructor(device='cuda')
        assert reconstructor.device == 'cuda'


class TestReconstructComplexSpec:
    """Test complex spectrogram reconstruction from different formats"""
    
    def test_reconstruct_complex_format(self, reconstructor, sample_complex):
        """Test reconstruction from COMPLEX format [real, imag]"""
        complex_spec = reconstructor._reconstruct_complex_spec(
            sample_complex, 
            DatasetFormat.COMPLEX
        )
        
        assert complex_spec.dtype == np.complex64 or complex_spec.dtype == np.complex128
        assert complex_spec.shape == (100, 1025)
        
        # Verify reconstruction accuracy
        real_reconstructed = complex_spec.real
        imag_reconstructed = complex_spec.imag
        
        np.testing.assert_allclose(real_reconstructed, sample_complex[..., 0], rtol=1e-5)
        np.testing.assert_allclose(imag_reconstructed, sample_complex[..., 1], rtol=1e-5)
    
    def test_reconstruct_magnitude_phase_format(self, reconstructor, sample_magnitude_phase):
        """Test reconstruction from MAGNITUDE_PHASE format [mag_db, phase]"""
        complex_spec = reconstructor._reconstruct_complex_spec(
            sample_magnitude_phase,
            DatasetFormat.MAGNITUDE_PHASE
        )
        
        assert complex_spec.dtype == np.complex64 or complex_spec.dtype == np.complex128
        assert complex_spec.shape == (100, 1025)
        
        # Verify magnitude (in dB) and phase
        magnitude_db_orig = sample_magnitude_phase[..., 0]
        phase_orig = sample_magnitude_phase[..., 1]
        
        magnitude_orig = 10 ** (magnitude_db_orig / 20.0)
        magnitude_reconstructed = np.abs(complex_spec)
        phase_reconstructed = np.angle(complex_spec)
        
        np.testing.assert_allclose(magnitude_reconstructed, magnitude_orig, rtol=1e-5)
        # Phase comparison with wrap-around handling
        phase_diff = np.angle(np.exp(1j * (phase_reconstructed - phase_orig)))
        np.testing.assert_allclose(phase_diff, 0, atol=1e-5)
    
    def test_reconstruct_magnitude_phase_trig_format(self, reconstructor, sample_magnitude_phase_trig):
        """Test reconstruction from MAGNITUDE_PHASE_TRIG format [mag_db, cos, sin]"""
        complex_spec = reconstructor._reconstruct_complex_spec(
            sample_magnitude_phase_trig,
            DatasetFormat.MAGNITUDE_PHASE_TRIG
        )
        
        assert complex_spec.dtype == np.complex64 or complex_spec.dtype == np.complex128
        assert complex_spec.shape == (100, 1025)
        
        # Verify magnitude and phase from cos/sin
        magnitude_db_orig = sample_magnitude_phase_trig[..., 0]
        cos_phase = sample_magnitude_phase_trig[..., 1]
        sin_phase = sample_magnitude_phase_trig[..., 2]
        
        magnitude_orig = 10 ** (magnitude_db_orig / 20.0)
        phase_orig = np.arctan2(sin_phase, cos_phase)
        
        magnitude_reconstructed = np.abs(complex_spec)
        phase_reconstructed = np.angle(complex_spec)
        
        np.testing.assert_allclose(magnitude_reconstructed, magnitude_orig, rtol=1e-5)
        phase_diff = np.angle(np.exp(1j * (phase_reconstructed - phase_orig)))
        np.testing.assert_allclose(phase_diff, 0, atol=1e-5)
    
    def test_reconstruct_magnitude_only_format(self, reconstructor, sample_magnitude_only):
        """Test reconstruction from MAGNITUDE_ONLY format [mag_db]"""
        complex_spec = reconstructor._reconstruct_complex_spec(
            sample_magnitude_only,
            DatasetFormat.MAGNITUDE_ONLY
        )
        
        assert complex_spec.dtype == np.complex64 or complex_spec.dtype == np.complex128
        assert complex_spec.shape == (100, 1025)
        
        # Verify magnitude, phase should be ~0
        magnitude_db_orig = sample_magnitude_only[..., 0]
        magnitude_orig = 10 ** (magnitude_db_orig / 20.0)
        
        magnitude_reconstructed = np.abs(complex_spec)
        phase_reconstructed = np.angle(complex_spec)
        
        np.testing.assert_allclose(magnitude_reconstructed, magnitude_orig, rtol=1e-5)
        np.testing.assert_allclose(phase_reconstructed, 0, atol=1e-5)


class TestReconstructFromModelOutput:
    """Test reconstruct_from_model_output method"""
    
    def test_reconstruct_magnitude_phase_trig_istft(
        self, 
        reconstructor, 
        sample_magnitude_phase_trig, 
        stft_params
    ):
        """Test ISTFT reconstruction from MAGNITUDE_PHASE_TRIG format"""
        audio = reconstructor.reconstruct_from_model_output(
            model_output=sample_magnitude_phase_trig,
            format=DatasetFormat.MAGNITUDE_PHASE_TRIG,
            stft_params=stft_params,
            method='istft'
        )
        
        assert isinstance(audio, AudioData)
        assert audio.sample_rate == 22050
        assert audio.n_channels == 1
        assert audio.waveform.ndim == 1
        assert len(audio.waveform) > 0
    
    def test_reconstruct_magnitude_only_auto_method(
        self,
        reconstructor,
        sample_magnitude_only,
        stft_params
    ):
        """Test auto method selection (should use griffin-lim for MAGNITUDE_ONLY)"""
        audio = reconstructor.reconstruct_from_model_output(
            model_output=sample_magnitude_only,
            format=DatasetFormat.MAGNITUDE_ONLY,
            stft_params=stft_params,
            method='auto'
        )
        
        assert isinstance(audio, AudioData)
        assert audio.sample_rate == 22050
        assert audio.n_channels == 1
    
    def test_reconstruct_complex_format(
        self,
        reconstructor,
        sample_complex,
        stft_params
    ):
        """Test reconstruction from COMPLEX format"""
        audio = reconstructor.reconstruct_from_model_output(
            model_output=sample_complex,
            format=DatasetFormat.COMPLEX,
            stft_params=stft_params,
            method='istft'
        )
        
        assert isinstance(audio, AudioData)
        assert audio.sample_rate == 22050
    
    def test_reconstruct_magnitude_phase_format(
        self,
        reconstructor,
        sample_magnitude_phase,
        stft_params
    ):
        """Test reconstruction from MAGNITUDE_PHASE format"""
        audio = reconstructor.reconstruct_from_model_output(
            model_output=sample_magnitude_phase,
            format=DatasetFormat.MAGNITUDE_PHASE,
            stft_params=stft_params,
            method='istft'
        )
        
        assert isinstance(audio, AudioData)
        assert audio.sample_rate == 22050
    
    def test_reconstruct_with_batch_dimension(
        self,
        reconstructor,
        sample_magnitude_phase_trig,
        stft_params
    ):
        """Test reconstruction with batch dimension (should use first sample)"""
        # Add batch dimension
        batch_input = sample_magnitude_phase_trig[np.newaxis, ...]
        assert batch_input.ndim == 4
        
        audio = reconstructor.reconstruct_from_model_output(
            model_output=batch_input,
            format=DatasetFormat.MAGNITUDE_PHASE_TRIG,
            stft_params=stft_params,
            method='istft'
        )
        
        assert isinstance(audio, AudioData)
        assert audio.sample_rate == 22050
    
    def test_reconstruct_with_original_length_padding_removal(
        self,
        reconstructor,
        sample_magnitude_phase_trig,
        stft_params
    ):
        """Test padding removal using original_length parameter"""
        # Reconstruct with padding
        audio_padded = reconstructor.reconstruct_from_model_output(
            model_output=sample_magnitude_phase_trig,
            format=DatasetFormat.MAGNITUDE_PHASE_TRIG,
            stft_params=stft_params,
            method='istft'
        )
        
        original_length = len(audio_padded.waveform) - 1000
        
        # Reconstruct with original_length trimming
        audio_trimmed = reconstructor.reconstruct_from_model_output(
            model_output=sample_magnitude_phase_trig,
            format=DatasetFormat.MAGNITUDE_PHASE_TRIG,
            stft_params=stft_params,
            method='istft',
            original_length=original_length
        )
        
        assert len(audio_trimmed.waveform) == original_length
        np.testing.assert_array_equal(
            audio_trimmed.waveform,
            audio_padded.waveform[:original_length]
        )


class TestReconstructFromDataset:
    """Test dataset loading and reconstruction"""
    
    def test_reconstruct_from_dataset(
        self,
        reconstructor,
        stft_params,
        sample_magnitude_phase_trig,
        tmp_path
    ):
        """Test loading and reconstructing from HDF5 dataset"""
        import h5py
        
        # Create a simple HDF5 dataset
        dataset_path = tmp_path / "test_dataset.h5"
        
        with h5py.File(dataset_path, 'w') as f:
            group = f.create_group('data')
            group.attrs['n_samples'] = 1
            
            sample_group = group.create_group('sample_000000')
            sample_group.create_dataset('data', data=sample_magnitude_phase_trig)
            sample_group.attrs['format'] = 'magnitude_phase_trig'
            sample_group.attrs['n_fft'] = 2048
            sample_group.attrs['hop_length'] = 512
            sample_group.attrs['win_length'] = 2048
            sample_group.attrs['window'] = 'hann'
            sample_group.attrs['sample_rate'] = 22050
            sample_group.attrs['original_length'] = 44100
        
        # Reconstruct from dataset
        audio = reconstructor.reconstruct_from_dataset(
            dataset_path=str(dataset_path),
            sample_index=0,
            split='data',
            method='istft'
        )
        
        assert isinstance(audio, AudioData)
        assert audio.sample_rate == 22050


class TestBatchReconstructFromDataset:
    """Test batch reconstruction from dataset"""
    
    def test_batch_reconstruct_from_dataset(
        self,
        reconstructor,
        sample_magnitude_phase_trig,
        tmp_path
    ):
        """Test batch reconstruction to WAV files"""
        import h5py
        
        # Create HDF5 dataset with multiple samples
        dataset_path = tmp_path / "test_dataset.h5"
        
        with h5py.File(dataset_path, 'w') as f:
            group = f.create_group('data')
            group.attrs['n_samples'] = 2
            
            for i in range(2):
                sample_group = group.create_group(f'sample_{i:06d}')
                sample_group.create_dataset('data', data=sample_magnitude_phase_trig)
                sample_group.attrs['format'] = 'magnitude_phase_trig'
                sample_group.attrs['n_fft'] = 2048
                sample_group.attrs['hop_length'] = 512
                sample_group.attrs['win_length'] = 2048
                sample_group.attrs['window'] = 'hann'
                sample_group.attrs['sample_rate'] = 22050
        
        # Batch reconstruct
        output_dir = tmp_path / "output"
        output_paths = reconstructor.batch_reconstruct_from_dataset(
            dataset_path=str(dataset_path),
            output_dir=str(output_dir),
            split='data',
            method='istft'
        )
        
        assert len(output_paths) == 2
        for path in output_paths:
            assert os.path.exists(path)
            assert path.endswith('.wav')


class TestInvalidInputs:
    """Test error handling for invalid inputs"""
    
    def test_invalid_format(self, reconstructor):
        """Test with invalid format"""
        data = np.random.randn(100, 1025, 1)
        
        with pytest.raises(ValueError):
            reconstructor._reconstruct_complex_spec(
                data,
                'invalid_format'
            )
    
    def test_missing_stft_params(self, reconstructor, sample_magnitude_phase_trig):
        """Test with missing STFT parameters"""
        incomplete_params = {
            'n_fft': 2048,
            'hop_length': 512
            # Missing win_length, window, sample_rate
        }
        
        with pytest.raises(KeyError):
            reconstructor.reconstruct_from_model_output(
                model_output=sample_magnitude_phase_trig,
                format=DatasetFormat.MAGNITUDE_PHASE_TRIG,
                stft_params=incomplete_params,
                method='istft'
            )
    
    def test_invalid_method(self, reconstructor, sample_magnitude_phase_trig, stft_params):
        """Test with invalid reconstruction method"""
        with pytest.raises(ValueError):
            reconstructor.reconstruct_from_model_output(
                model_output=sample_magnitude_phase_trig,
                format=DatasetFormat.MAGNITUDE_PHASE_TRIG,
                stft_params=stft_params,
                method='invalid_method'
            )
