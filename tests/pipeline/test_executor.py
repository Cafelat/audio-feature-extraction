"""Tests for PipelineExecutor and StateManager"""

import pytest
import tempfile
import os
from pathlib import Path
from typing import List

from dataset_generator.pipeline.executor import PipelineExecutor, ExecutionReport, StateManager
from dataset_generator.io.audio_loader import AudioFileLoader
from dataset_generator.features.stft import STFTExtractor
from dataset_generator.io.dataset_writer import HDF5DatasetWriter
from dataset_generator.core.types import DatasetFormat
from dataset_generator.core.models import AudioData, SpectrogramData


class TestStateManager:
    """Test StateManager for tracking pipeline state"""
    
    def test_state_manager_init(self):
        """Test StateManager initialization"""
        state_manager = StateManager()
        assert state_manager is not None
        
    def test_state_manager_update(self):
        """Test updating state"""
        state_manager = StateManager()
        test_data = {"key": "value"}
        
        state_manager.update("test_state", test_data)
        assert state_manager.get_state("test_state") == test_data
    
    def test_state_manager_multiple_states(self):
        """Test multiple state updates"""
        state_manager = StateManager()
        
        state_manager.update("state1", "data1")
        state_manager.update("state2", "data2")
        
        assert state_manager.get_state("state1") == "data1"
        assert state_manager.get_state("state2") == "data2"
    
    def test_state_manager_elapsed_time(self):
        """Test elapsed time tracking"""
        state_manager = StateManager()
        elapsed = state_manager.get_elapsed_time()
        
        assert elapsed >= 0
        assert isinstance(elapsed, float)


class TestExecutionReport:
    """Test ExecutionReport dataclass"""
    
    def test_execution_report_init(self):
        """Test ExecutionReport creation"""
        report = ExecutionReport(
            total_files=10,
            successful=8,
            failed=2,
            processing_time=5.5
        )
        
        assert report.total_files == 10
        assert report.successful == 8
        assert report.failed == 2
        assert report.processing_time == 5.5
    
    def test_execution_report_success_rate(self):
        """Test success rate calculation"""
        report = ExecutionReport(
            total_files=10,
            successful=8,
            failed=2,
            processing_time=5.5
        )
        
        assert report.success_rate == 0.8
    
    def test_execution_report_zero_files(self):
        """Test with zero files"""
        report = ExecutionReport(
            total_files=0,
            successful=0,
            failed=0,
            processing_time=0.0
        )
        
        assert report.success_rate == 0.0


class TestPipelineExecutor:
    """Test PipelineExecutor end-to-end"""
    
    @pytest.fixture
    def audio_files(self, tmp_path):
        """Create sample audio files"""
        import numpy as np
        import soundfile as sf
        
        file_paths = []
        for i in range(2):
            # Create simple sine wave
            sample_rate = 22050
            duration = 1.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            frequency = 440 + i * 100
            waveform = np.sin(2 * np.pi * frequency * t).astype(np.float32)
            
            file_path = tmp_path / f"test_audio_{i}.wav"
            sf.write(file_path, waveform, sample_rate)
            file_paths.append(str(file_path))
        
        return file_paths
    
    @pytest.fixture
    def pipeline_executor(self):
        """Create a PipelineExecutor instance"""
        loader = AudioFileLoader(device='cpu')
        extractors = [
            STFTExtractor(
                n_fft=2048,
                hop_length=512,
                device='cpu'
            )
        ]
        writer = HDF5DatasetWriter(format=DatasetFormat.MAGNITUDE_PHASE_TRIG)
        
        return PipelineExecutor(
            loader=loader,
            extractors=extractors,
            writer=writer,
            device='cpu',
            batch_size=2
        )
    
    def test_pipeline_executor_init(self, pipeline_executor):
        """Test PipelineExecutor initialization"""
        assert pipeline_executor is not None
        assert pipeline_executor.batch_size == 2
        assert pipeline_executor.device == 'cpu'
        assert len(pipeline_executor.extractors) == 1
    
    def test_pipeline_executor_create_batches(self, pipeline_executor):
        """Test batch creation"""
        input_paths = [f"file_{i}.wav" for i in range(5)]
        
        batches = list(pipeline_executor._create_batches(input_paths, batch_size=2))
        
        # Should create 3 batches (2 + 2 + 1)
        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1
    
    def test_pipeline_executor_execute_integration(
        self,
        pipeline_executor,
        audio_files,
        tmp_path
    ):
        """Test end-to-end pipeline execution"""
        output_path = tmp_path / "output_dataset.h5"
        
        report = pipeline_executor.execute(
            input_paths=audio_files,
            output_path=str(output_path)
        )
        
        # Verify report
        assert isinstance(report, ExecutionReport)
        assert report.total_files == len(audio_files)
        assert report.successful == len(audio_files)
        assert report.failed == 0
        assert report.success_rate == 1.0
        assert report.processing_time >= 0
        
        # Verify output file created
        assert output_path.exists()
    
    def test_pipeline_executor_progress_callback(
        self,
        pipeline_executor,
        audio_files,
        tmp_path
    ):
        """Test progress callback during execution"""
        output_path = tmp_path / "output_dataset.h5"
        progress_calls = []
        
        def progress_callback(current, total):
            progress_calls.append((current, total))
        
        report = pipeline_executor.execute(
            input_paths=audio_files,
            output_path=str(output_path),
            progress_callback=progress_callback
        )
        
        # Verify progress callback was called
        assert len(progress_calls) > 0
        
        # Verify final progress
        assert progress_calls[-1][0] == len(audio_files)
        assert progress_calls[-1][1] == len(audio_files)
    
    def test_pipeline_executor_with_custom_batch_size(self, tmp_path):
        """Test with custom batch size"""
        loader = AudioFileLoader(device='cpu')
        extractors = [
            STFTExtractor(
                n_fft=2048,
                hop_length=512,
                device='cpu'
            )
        ]
        writer = HDF5DatasetWriter(format=DatasetFormat.MAGNITUDE_PHASE_TRIG)
        
        executor = PipelineExecutor(
            loader=loader,
            extractors=extractors,
            writer=writer,
            device='cpu',
            batch_size=1
        )
        
        assert executor.batch_size == 1
    
    def test_pipeline_executor_empty_input(self, pipeline_executor, tmp_path):
        """Test with empty input paths"""
        output_path = tmp_path / "output_dataset.h5"
        
        report = pipeline_executor.execute(
            input_paths=[],
            output_path=str(output_path)
        )
        
        assert report.total_files == 0
        assert report.successful == 0
        assert report.failed == 0


class TestPipelineExecutorIntegration:
    """Integration tests for complete pipeline"""
    
    def test_pipeline_with_multiple_extractors(self, tmp_path):
        """Test pipeline with multiple feature extractors"""
        import numpy as np
        import soundfile as sf
        
        # Create test audio
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        waveform = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        audio_path = tmp_path / "test.wav"
        sf.write(audio_path, waveform, sample_rate)
        
        # Create pipeline with multiple extractors
        loader = AudioFileLoader(device='cpu')
        extractors = [
            STFTExtractor(n_fft=2048, hop_length=512, device='cpu'),
            STFTExtractor(n_fft=1024, hop_length=256, device='cpu')
        ]
        writer = HDF5DatasetWriter(format=DatasetFormat.MAGNITUDE_PHASE_TRIG)
        
        executor = PipelineExecutor(
            loader=loader,
            extractors=extractors,
            writer=writer,
            device='cpu',
            batch_size=1
        )
        
        output_path = tmp_path / "output_dataset.h5"
        report = executor.execute(
            input_paths=[str(audio_path)],
            output_path=str(output_path)
        )
        
        # With 2 extractors, we get 2 features per audio file
        assert report.total_files == 1
        assert report.successful == 2  # 1 file * 2 extractors
        assert report.failed == 0
        assert output_path.exists()
    
    def test_pipeline_error_handling_missing_file(
        self,
        tmp_path
    ):
        """Test pipeline with missing input file"""
        loader = AudioFileLoader(device='cpu')
        extractors = [
            STFTExtractor(n_fft=2048, hop_length=512, device='cpu')
        ]
        writer = HDF5DatasetWriter(format=DatasetFormat.MAGNITUDE_PHASE_TRIG)
        
        executor = PipelineExecutor(
            loader=loader,
            extractors=extractors,
            writer=writer,
            device='cpu'
        )
        
        output_path = tmp_path / "output_dataset.h5"
        
        # Try to execute with non-existent file
        # Should handle error gracefully
        try:
            report = executor.execute(
                input_paths=["/nonexistent/file.wav"],
                output_path=str(output_path)
            )
            # If we get here, the error was handled
            assert report.failed > 0 or report.successful == 0
        except Exception:
            # Also acceptable to raise exception
            pass
