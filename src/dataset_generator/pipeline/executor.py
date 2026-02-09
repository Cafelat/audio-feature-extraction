"""Pipeline executor for audio processing"""

import time
from typing import List, Callable, Optional, Dict, Any
from dataclasses import dataclass

from ..core.types import AudioLoader, FeatureExtractor, DatasetWriter


@dataclass
class ExecutionReport:
    """Report of pipeline execution"""
    total_files: int
    successful: int
    failed: int
    processing_time: float
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)"""
        if self.total_files == 0:
            return 0.0
        return self.successful / self.total_files


class StateManager:
    """Pipeline state management
    
    Tracks state transitions and processing time in the pipeline.
    """
    
    def __init__(self):
        """Initialize StateManager"""
        self.states: Dict[str, Any] = {}
        self.start_time = time.time()
    
    def update(self, state_name: str, data: Any) -> None:
        """Update pipeline state
        
        Args:
            state_name: Name of the state
            data: Data associated with state
        """
        self.states[state_name] = data
    
    def get_state(self, state_name: str) -> Optional[Any]:
        """Get pipeline state
        
        Args:
            state_name: Name of the state
            
        Returns:
            Data associated with state, or None if not found
        """
        return self.states.get(state_name)
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since initialization
        
        Returns:
            Elapsed time in seconds
        """
        return time.time() - self.start_time
    
    def reset(self) -> None:
        """Reset state manager"""
        self.states.clear()
        self.start_time = time.time()


class PipelineExecutor:
    """バッチ処理パイプライン実行器
    
    ローダー、抽出器、ライターを統合してEnd-to-Endのパイプラインを実行。
    """
    
    def __init__(
        self,
        loader: AudioLoader,
        extractors: List[FeatureExtractor],
        writer: DatasetWriter,
        device: str = 'cpu',
        batch_size: int = 16,
        num_workers: int = 4
    ):
        """Initialize PipelineExecutor
        
        Args:
            loader: Audio loader instance
            extractors: List of feature extractors
            writer: Dataset writer instance
            device: Computation device ('cpu' or 'cuda')
            batch_size: Batch size for processing
            num_workers: Number of worker threads (for future parallel processing)
        """
        self.loader = loader
        self.extractors = extractors
        self.writer = writer
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.state_manager = StateManager()
    
    def execute(
        self,
        input_paths: List[str],
        output_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> ExecutionReport:
        """パイプライン実行
        
        Args:
            input_paths: 入力音声ファイルパス一覧
            output_path: 出力データセットパス
            progress_callback: 進捗コールバック (current, total)
            
        Returns:
            ExecutionReport: 実行結果レポート
        """
        start_time = time.time()
        results = []
        failed_count = 0
        
        # バッチ処理
        for batch_paths in self._create_batches(input_paths):
            try:
                # 1. 読み込み
                audio_batch = self.loader.load_batch(batch_paths)
                self.state_manager.update('loaded', audio_batch)
                
                # 2. 特徴量抽出
                features_batch = []
                for audio in audio_batch:
                    for extractor in self.extractors:
                        feature = extractor.extract(audio)
                        features_batch.append(feature)
                
                self.state_manager.update('extracted', features_batch)
                
                # 3. 書き込み
                self.writer.write(features_batch, output_path)
                self.state_manager.update('saved', features_batch)
                
                results.extend(features_batch)
            
            except Exception as e:
                # Handle errors gracefully
                failed_count += len(batch_paths)
                if progress_callback:
                    progress_callback(len(results), len(input_paths))
                continue
            
            # Progress callback
            if progress_callback:
                progress_callback(len(results), len(input_paths))
        
        processing_time = time.time() - start_time
        
        return ExecutionReport(
            total_files=len(input_paths),
            successful=len(results),
            failed=failed_count,
            processing_time=processing_time
        )
    
    def _create_batches(
        self,
        paths: List[str],
        batch_size: Optional[int] = None
    ):
        """Create batches from input paths
        
        Args:
            paths: List of input file paths
            batch_size: Batch size (uses self.batch_size if None)
            
        Yields:
            List of paths for each batch
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        for i in range(0, len(paths), batch_size):
            yield paths[i:i + batch_size]
