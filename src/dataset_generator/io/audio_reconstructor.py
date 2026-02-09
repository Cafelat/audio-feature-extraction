"""AudioReconstructor class for reconstructing audio from model outputs"""

import numpy as np
import torch
from typing import Dict, Any, Optional, List
import os

from dataset_generator.core.models import AudioData, SpectrogramData
from dataset_generator.core.types import DatasetFormat
from dataset_generator.core.conversions import TensorConverter
from dataset_generator.transforms.inverse import ISTFTReconstructor, GriffinLimReconstructor


class AudioReconstructor:
    """モデル出力から音声を復元
    
    学習済みモデルの出力（予測されたスペクトログラム）から
    音声ファイルに戻す。
    """
    
    def __init__(self, device: str = 'cpu'):
        """Initialize AudioReconstructor
        
        Args:
            device: Computation device ('cpu' or 'cuda')
        """
        self.device = device
        self.istft_reconstructor = ISTFTReconstructor(device=device)
        self.griffin_lim_reconstructor = GriffinLimReconstructor(device=device)
        self.converter = TensorConverter()
    
    def reconstruct_from_model_output(
        self,
        model_output: np.ndarray | torch.Tensor,
        format: DatasetFormat,
        stft_params: Dict[str, Any],
        method: str = 'auto',
        original_length: Optional[int] = None
    ) -> AudioData:
        """モデル出力から音声を復元
        
        Args:
            model_output: モデルの予測出力 (time, freq, channels) or (batch, time, freq, channels)
            format: データ形式
            stft_params: STFT復元パラメータ (n_fft, hop_length, win_length, window, sample_rate)
            method: 復元方法（'auto'は形式に応じて自動選択）
            original_length: 元の音声長（パディング除去用）
            
        Returns:
            復元された音声データ
        """
        # バッチ次元があれば最初のサンプルのみ処理
        if isinstance(model_output, torch.Tensor):
            if model_output.ndim == 4:
                model_output = model_output[0]
            model_output = self.converter.to_numpy(model_output)
        elif isinstance(model_output, np.ndarray):
            if model_output.ndim == 4:
                model_output = model_output[0]
        
        # 形式に応じて複素数スペクトログラムを復元
        complex_spec = self._reconstruct_complex_spec(model_output, format)
        
        # 復元方法の自動選択
        if method == 'auto':
            if format == DatasetFormat.MAGNITUDE_ONLY:
                method = 'griffin-lim'
            else:
                method = 'istft'
        
        # SpectrogramData作成
        spec_data = SpectrogramData(
            complex_spec=torch.from_numpy(complex_spec).to(self.device),
            magnitude_db=torch.abs(torch.from_numpy(complex_spec)).to(self.device),  # 仮
            phase=torch.angle(torch.from_numpy(complex_spec)).to(self.device),  # 仮
            n_fft=stft_params['n_fft'],
            hop_length=stft_params['hop_length'],
            win_length=stft_params['win_length'],
            window=stft_params['window'],
            sample_rate=stft_params['sample_rate'],
            metadata={'reconstructed_from_model': True}
        )
        
        # ISTFT or Griffin-Lim
        if method == 'istft':
            audio = self.istft_reconstructor.reconstruct(spec_data)
        elif method == 'griffin-lim':
            audio = self.griffin_lim_reconstructor.reconstruct(spec_data)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # 元の長さに調整（パディング除去）
        if original_length is not None:
            current_length = audio.waveform.shape[-1]
            if current_length > original_length:
                audio.waveform = audio.waveform[..., :original_length]
                audio.duration = original_length / stft_params['sample_rate']
        
        return audio
    
    def _reconstruct_complex_spec(
        self,
        model_output: np.ndarray,
        format: DatasetFormat | str
    ) -> np.ndarray:
        """モデル出力から複素数スペクトログラムを復元
        
        Args:
            model_output: (time, freq, channels)
            format: データ形式
            
        Returns:
            complex_spec: (time, freq) - 複素数型
        """
        # Convert string to DatasetFormat if needed
        if isinstance(format, str):
            format = DatasetFormat(format)
        
        if format == DatasetFormat.COMPLEX:
            # [real, imag] から複素数復元
            real = model_output[..., 0]
            imag = model_output[..., 1]
            complex_spec = real + 1j * imag
        
        elif format == DatasetFormat.MAGNITUDE_PHASE:
            # [mag_db, phase] から複素数復元
            mag_db = model_output[..., 0]
            phase = model_output[..., 1]
            
            magnitude = 10 ** (mag_db / 20.0)
            complex_spec = magnitude * np.exp(1j * phase)
        
        elif format == DatasetFormat.MAGNITUDE_PHASE_TRIG:
            # [mag_db, cos, sin] から複素数復元
            mag_db = model_output[..., 0]
            cos_phase = model_output[..., 1]
            sin_phase = model_output[..., 2]
            
            magnitude = 10 ** (mag_db / 20.0)
            phase = np.arctan2(sin_phase, cos_phase)
            complex_spec = magnitude * np.exp(1j * phase)
        
        elif format == DatasetFormat.MAGNITUDE_ONLY:
            # [mag_db] のみ → 振幅のみ（位相は0で初期化、Griffin-Lim用）
            mag_db = model_output[..., 0]
            magnitude = 10 ** (mag_db / 20.0)
            complex_spec = magnitude.astype(np.complex64)  # 位相0
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        return complex_spec
    
    def reconstruct_from_dataset(
        self,
        dataset_path: str,
        sample_index: int,
        split: str = 'data',
        method: str = 'auto'
    ) -> AudioData:
        """データセットから音声を復元（検証・デバッグ用）
        
        Args:
            dataset_path: HDF5データセットパス
            sample_index: サンプルインデックス
            split: データ分割名
            method: 復元方法
            
        Returns:
            復元された音声データ
        """
        import h5py
        
        with h5py.File(dataset_path, 'r') as f:
            group = f[split]
            sample_key = f'sample_{sample_index:06d}'
            sample = group[sample_key]
            
            # メタデータ読み込み
            format_str = sample.attrs.get('format', 'magnitude_phase_trig')
            format = DatasetFormat(format_str)
            
            stft_params = {
                'n_fft': sample.attrs['n_fft'],
                'hop_length': sample.attrs['hop_length'],
                'win_length': sample.attrs['win_length'],
                'window': sample.attrs['window'],
                'sample_rate': sample.attrs['sample_rate']
            }
            
            # データ読み込み
            data = sample['data'][:]
            
            # 元の音声長
            original_length = sample.attrs.get('original_length', None)
            
            # 復元
            return self.reconstruct_from_model_output(
                model_output=data,
                format=format,
                stft_params=stft_params,
                method=method,
                original_length=original_length
            )
    
    def batch_reconstruct_from_dataset(
        self,
        dataset_path: str,
        output_dir: str,
        split: str = 'data',
        method: str = 'auto'
    ) -> List[str]:
        """データセット全体を音声ファイルに復元（検証用）
        
        Args:
            dataset_path: HDF5データセットパス
            output_dir: 出力ディレクトリ
            split: データ分割名
            method: 復元方法
            
        Returns:
            出力ファイルパスのリスト
        """
        import h5py
        import soundfile as sf
        
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        
        with h5py.File(dataset_path, 'r') as f:
            group = f[split]
            n_samples = group.attrs['n_samples']
            
            for i in range(n_samples):
                # 音声復元
                audio = self.reconstruct_from_dataset(
                    dataset_path, i, split, method
                )
                
                # ファイル保存
                output_path = os.path.join(output_dir, f'reconstructed_{i:06d}.wav')
                waveform_np = self.converter.to_numpy(audio.waveform)
                
                sf.write(output_path, waveform_np, audio.sample_rate)
                output_paths.append(output_path)
        
        return output_paths
