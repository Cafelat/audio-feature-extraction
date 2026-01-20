# 設計書: 音声特徴量抽出モジュール及びデータセットジェネレータ

## 1. システムアーキテクチャ

### 1.1 アーキテクチャ概要

本システムは、**レイヤードアーキテクチャ**と**パイプラインパターン**を組み合わせた設計を採用します。各処理段階を独立したモジュールとして実装し、直感的な状態管理とデータフローを実現します。

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  (CLI, API, Configuration Management)                        │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Pipeline Orchestration Layer               │
│  (Pipeline Builder, State Manager, Real-time Processor)      │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     Processing Layer                         │
│  ┌───────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐ │
│  │  Loader   │→│ Extractor│→│Transformer│→│   Writer  │ │
│  └───────────┘  └──────────┘  └──────────┘  └───────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                       Core Layer                             │
│  (Data Models, Audio Operations, GPU Acceleration)           │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 設計原則

1. **状態の明示性**: 各処理段階の入出力データ構造を型付きデータクラスで定義
2. **単一責任の原則**: 各モジュールは1つの明確な責任を持つ
3. **依存性の逆転**: インターフェース（Protocol）を通じた疎結合
4. **オープン・クローズドの原則**: 拡張に対して開いており、修正に対して閉じている
5. **リアルタイム対応**: ストリーミング処理とバッチ処理の両方をサポート

## 2. モジュール構成

### 2.1 ディレクトリ構造

```
src/dataset_generator/
├── __init__.py
├── core/                      # コアデータ構造・基本操作
│   ├── __init__.py
│   ├── models.py             # データモデル定義
│   ├── types.py              # 型定義・Protocol
│   ├── audio_ops.py          # 音声基本操作
│   └── device.py             # GPU/CPUデバイス管理
│
├── io/                        # 入出力モジュール
│   ├── __init__.py
│   ├── audio_loader.py       # 音声ファイル読み込み
│   ├── stream_loader.py      # ストリーミング音声読み込み
│   └── dataset_writer.py     # データセット書き込み
│
├── features/                  # 特徴量抽出モジュール
│   ├── __init__.py
│   ├── base.py               # 抽出器基底クラス
│   ├── stft.py               # STFT抽出器
│   ├── mel.py                # メルスペクトログラム抽出器
│   ├── mfcc.py               # MFCC抽出器
│   └── registry.py           # カスタム抽出器レジストリ
│
├── transforms/                # 変換・前処理モジュール
│   ├── __init__.py
│   ├── preprocessor.py       # 前処理（正規化、トリミング等）
│   ├── augmentation.py       # データ拡張
│   └── inverse.py            # 逆変換（ISTFT, Griffin-Lim）
│
├── pipeline/                  # パイプライン制御
│   ├── __init__.py
│   ├── builder.py            # パイプライン構築
│   ├── executor.py           # バッチ実行
│   ├── stream_executor.py    # リアルタイム実行
│   └── state_manager.py      # 状態管理
│
├── config/                    # 設定管理
│   ├── __init__.py
│   ├── schema.py             # 設定スキーマ
│   └── loader.py             # YAML/JSON設定読み込み
│
├── cli/                       # コマンドラインインターフェース
│   ├── __init__.py
│   └── main.py               # CLIエントリーポイント
│
└── utils/                     # ユーティリティ
    ├── __init__.py
    ├── logging.py            # ロギング
    ├── metrics.py            # メトリクス計測
    └── visualization.py      # 可視化（デバッグ用）
```

## 3. コアデータモデル

### 3.1 主要データクラス

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import torch

@dataclass
class AudioData:
    """音声波形データ"""
    waveform: np.ndarray | torch.Tensor  # Shape: (n_samples,) or (n_channels, n_samples)
    sample_rate: int
    n_channels: int
    duration: float
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """データ形状・型の検証"""
        # 波形データの次元チェック
        if isinstance(self.waveform, np.ndarray):
            ndim = self.waveform.ndim
        elif isinstance(self.waveform, torch.Tensor):
            ndim = self.waveform.ndim
        else:
            raise TypeError(f"waveform must be np.ndarray or torch.Tensor, got {type(self.waveform)}")
        
        if ndim not in [1, 2]:
            raise ValueError(f"waveform must be 1D or 2D, got {ndim}D")
        
        # チャンネル数の整合性チェック
        if ndim == 1:
            if self.n_channels != 1:
                raise ValueError(f"1D waveform requires n_channels=1, got {self.n_channels}")
        elif ndim == 2:
            actual_channels = self.waveform.shape[0]
            if actual_channels != self.n_channels:
                raise ValueError(f"n_channels mismatch: expected {self.n_channels}, got {actual_channels}")
        
        # サンプリングレートの妥当性チェック
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        
        # 継続時間の計算検証
        n_samples = self.waveform.shape[-1]  # 最後の次元がサンプル数
        expected_duration = n_samples / self.sample_rate
        if abs(self.duration - expected_duration) > 1e-3:  # 1ms以内の誤差を許容
            import warnings
            warnings.warn(f"Duration mismatch: provided {self.duration:.3f}s, "
                        f"calculated {expected_duration:.3f}s from waveform")

@dataclass
class SpectrogramData:
    """スペクトログラムデータ（STFT結果）"""
    # 複素数スペクトログラム
    complex_spec: np.ndarray | torch.Tensor  # Shape: (n_frames, n_freq_bins)
    
    # 振幅スペクトログラム（dBスケール）
    magnitude_db: np.ndarray | torch.Tensor  # Shape: (n_frames, n_freq_bins)
    
    # 位相スペクトログラム（ラジアン）
    phase: np.ndarray | torch.Tensor  # Shape: (n_frames, n_freq_bins)
    
    # パラメータ
    n_fft: int
    hop_length: int
    win_length: int
    window: str
    sample_rate: int
    
    # メタデータ
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """データ形状・型の検証"""
        # 形状の一致確認
        if self.magnitude_db.shape != self.phase.shape:
            raise ValueError(f"magnitude_db and phase shape mismatch: "
                           f"{self.magnitude_db.shape} vs {self.phase.shape}")
        
        if self.complex_spec.shape != self.magnitude_db.shape:
            raise ValueError(f"complex_spec and magnitude_db shape mismatch: "
                           f"{self.complex_spec.shape} vs {self.magnitude_db.shape}")
        
        # 周波数ビン数の検証
        expected_freq_bins = self.n_fft // 2 + 1
        actual_freq_bins = self.magnitude_db.shape[1]
        if actual_freq_bins != expected_freq_bins:
            raise ValueError(f"Frequency bins mismatch: expected {expected_freq_bins} "
                           f"(n_fft={self.n_fft}), got {actual_freq_bins}")
        
        # complex_spec の型検証
        if isinstance(self.complex_spec, np.ndarray):
            if not np.iscomplexobj(self.complex_spec):
                raise TypeError("complex_spec must be complex dtype")
        elif isinstance(self.complex_spec, torch.Tensor):
            if not self.complex_spec.is_complex():
                raise TypeError("complex_spec must be complex dtype")
        
        # パラメータの妥当性検証
        if self.n_fft <= 0 or self.hop_length <= 0 or self.win_length <= 0:
            raise ValueError("n_fft, hop_length, win_length must be positive")
        
        if self.hop_length > self.n_fft:
            raise ValueError(f"hop_length ({self.hop_length}) cannot exceed n_fft ({self.n_fft})")
    
    def to_channels(self) -> np.ndarray | torch.Tensor:
        """(n_frames, n_freq_bins, 2) 形状に変換: [magnitude_db, phase]"""
        if isinstance(self.magnitude_db, np.ndarray):
            return np.stack([self.magnitude_db, self.phase], axis=-1)
        else:  # torch.Tensor
            return torch.stack([self.magnitude_db, self.phase], dim=-1)
    
    def save_params(self) -> Dict[str, Any]:
        """逆変換用パラメータを保存"""
        return {
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'win_length': self.win_length,
            'window': self.window,
            'sample_rate': self.sample_rate
        }

@dataclass
class MelSpectrogramData:
    """メルスペクトログラムデータ"""
    mel_spec_db: np.ndarray | torch.Tensor  # Shape: (n_frames, n_mels)
    n_mels: int
    fmin: float
    fmax: float
    sample_rate: int
    stft_params: Dict[str, Any]  # STFT parameters
    metadata: Dict[str, Any]

@dataclass
class FeatureData:
    """汎用特徴量データ"""
    features: np.ndarray | torch.Tensor
    feature_type: str  # 'mfcc', 'chroma', 'spectral_centroid' etc.
    shape: tuple
    params: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class ProcessingState:
    """処理パイプラインの状態"""
    stage: str  # 'loaded', 'extracted', 'transformed', 'saved'
    data: AudioData | SpectrogramData | MelSpectrogramData | FeatureData
    timestamp: float
    device: str  # 'cpu' or 'cuda:0'
    
    def describe(self) -> str:
        """状態の人間可読な説明"""
        pass
```

### 3.2 インターフェース定義（Protocol）

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class AudioLoader(Protocol):
    """音声読み込みインターフェース"""
    def load(self, path: str, **kwargs) -> AudioData:
        """音声ファイルを読み込む"""
        ...
    
    def load_batch(self, paths: list[str], **kwargs) -> list[AudioData]:
        """複数の音声ファイルをバッチ読み込み"""
        ...

@runtime_checkable
class FeatureExtractor(Protocol):
    """特徴量抽出インターフェース"""
    def extract(self, audio: AudioData) -> SpectrogramData | MelSpectrogramData | FeatureData:
        """特徴量を抽出"""
        ...
    
    def get_params(self) -> Dict[str, Any]:
        """抽出パラメータを取得"""
        ...

@runtime_checkable
class InverseTransform(Protocol):
    """逆変換インターフェース"""
    def reconstruct(
        self, 
        spec: SpectrogramData | MelSpectrogramData
    ) -> AudioData:
        """スペクトログラムから音声を再構成"""
        ...

@runtime_checkable
class DatasetWriter(Protocol):
    """データセット書き込みインターフェース"""
    def write(
        self, 
        data: list[SpectrogramData | MelSpectrogramData | FeatureData],
        output_path: str,
        **kwargs
    ) -> None:
        """データセットを書き込む"""
        ...
```

### 3.3 NumPy/PyTorch変換ポリシー

本システムでは、**内部処理はPyTorch統一、I/O境界でNumPy変換**の方針を採用します。

#### 変換方針

```python
"""
NumPy ↔ PyTorch 変換ルール

【基本方針】
- 内部処理（特徴量抽出、GPU演算）: PyTorch Tensor
- I/O境界（ファイル読み込み、保存）: NumPy ndarray
- データクラス: 両方を受け入れ可能（Union型）
- 変換は明示的に行い、自動変換は最小限に

【変換タイミング】
1. 入力時: NumPy → PyTorch (AudioLoader)
2. 処理中: PyTorch のみ
3. 出力時: PyTorch → NumPy (DatasetWriter)
"""

# core/conversions.py
class TensorConverter:
    """NumPy ↔ PyTorch 変換ユーティリティ"""
    
    @staticmethod
    def to_torch(
        array: np.ndarray | torch.Tensor,
        device: str = 'cpu',
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """NumPy配列をPyTorch Tensorに変換
        
        Args:
            array: 入力配列（NumPy or PyTorch）
            device: 転送先デバイス ('cpu', 'cuda', 'cuda:0' 等)
            dtype: 変換後のデータ型（Noneの場合は元の型を保持）
            
        Returns:
            PyTorch Tensor
        """
        if isinstance(array, torch.Tensor):
            tensor = array
        elif isinstance(array, np.ndarray):
            # NumPy → PyTorch
            tensor = torch.from_numpy(array)
        else:
            raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(array)}")
        
        # デバイス転送
        tensor = tensor.to(device)
        
        # データ型変換（指定がある場合）
        if dtype is not None:
            tensor = tensor.to(dtype)
        
        return tensor
    
    @staticmethod
    def to_numpy(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
        """PyTorch TensorをNumPy配列に変換
        
        Args:
            tensor: 入力Tensor（PyTorch or NumPy）
            
        Returns:
            NumPy ndarray
        """
        if isinstance(tensor, np.ndarray):
            return tensor
        elif isinstance(tensor, torch.Tensor):
            # GPU上のTensorの場合、まずCPUに転送
            if tensor.is_cuda:
                tensor = tensor.cpu()
            
            # 勾配追跡を解除
            if tensor.requires_grad:
                tensor = tensor.detach()
            
            # NumPyに変換
            return tensor.numpy()
        else:
            raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(tensor)}")
    
    @staticmethod
    def ensure_torch(
        data: Any,
        device: str = 'cpu'
    ) -> Any:
        """データクラス内の全配列をPyTorchに変換
        
        AudioData, SpectrogramData等のデータクラスに適用
        """
        if isinstance(data, AudioData):
            return AudioData(
                waveform=TensorConverter.to_torch(data.waveform, device),
                sample_rate=data.sample_rate,
                n_channels=data.n_channels,
                duration=data.duration,
                metadata=data.metadata
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
                metadata=data.metadata
            )
        else:
            return data
    
    @staticmethod
    def ensure_numpy(data: Any) -> Any:
        """データクラス内の全Tensorをnumpyに変換"""
        if isinstance(data, AudioData):
            return AudioData(
                waveform=TensorConverter.to_numpy(data.waveform),
                sample_rate=data.sample_rate,
                n_channels=data.n_channels,
                duration=data.duration,
                metadata=data.metadata
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
                metadata=data.metadata
            )
        else:
            return data


# 使用例
class AudioFileLoader:
    """音声ファイル読み込み（NumPy → PyTorch変換）"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.converter = TensorConverter()
    
    def load(self, path: str) -> AudioData:
        """音声ファイルを読み込み、PyTorch Tensorに変換"""
        import soundfile as sf
        
        # soundfile は NumPy で読み込む
        waveform_np, sample_rate = sf.read(path)
        
        # NumPy → PyTorch
        waveform = self.converter.to_torch(waveform_np, device=self.device)
        
        # AudioData 作成（内部はPyTorch）
        return AudioData(
            waveform=waveform,
            sample_rate=sample_rate,
            n_channels=1 if waveform.ndim == 1 else waveform.shape[0],
            duration=len(waveform_np) / sample_rate,
            metadata={'file_path': path, 'device': self.device}
        )


class HDF5Writer:
    """データセット書き込み（PyTorch → NumPy変換）"""
    
    def __init__(self):
        self.converter = TensorConverter()
    
    def write(self, data: list[SpectrogramData], output_path: str):
        """PyTorch TensorをNumPyに変換してHDF5保存"""
        import h5py
        
        with h5py.File(output_path, 'w') as f:
            for i, spec in enumerate(data):
                # PyTorch → NumPy
                spec_np = self.converter.ensure_numpy(spec)
                
                # HDF5に書き込み
                group = f.create_group(f'sample_{i}')
                group.create_dataset('magnitude_db', data=spec_np.magnitude_db)
                group.create_dataset('phase', data=spec_np.phase)
                group.attrs['n_fft'] = spec_np.n_fft
                group.attrs['hop_length'] = spec_np.hop_length
```

**変換ポリシーまとめ**:

| レイヤー | データ型 | 変換タイミング |
|---------|---------|---------------|
| I/O (入力) | NumPy → PyTorch | AudioLoader.load() |
| Processing | PyTorch | 変換なし（統一） |
| I/O (出力) | PyTorch → NumPy | DatasetWriter.write() |
| データクラス | Union型受入 | 必要に応じて変換 |

**メリット**:
- GPU処理の効率化（PyTorch統一）
- ファイルI/Oの互換性（NumPy）
- 明示的な変換で予期しない挙動を防止

**デメリット回避**:
- 変換オーバーヘッド → I/O境界のみで変換（処理中は変換なし）
- メモリコピー → 可能な限り `torch.from_numpy()` (共有メモリ)
```

## 4. 主要モジュール詳細設計

### 4.1 STFT抽出器（features/stft.py）

```python
class STFTExtractor:
    """STFT特徴量抽出器
    
    振幅・位相スペクトログラムを抽出し、dBスケール変換を行う。
    GPU加速対応。
    """
    
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: str = 'hann',
        center: bool = True,
        device: str = 'cpu'
    ):
        """
        Args:
            n_fft: FFTサイズ
            hop_length: ホップ長（サンプル）
            win_length: 窓長（Noneの場合はn_fftと同じ）
            window: 窓関数（'hann', 'hamming', 'blackman'）
            center: 中央パディング有効化
            device: 'cpu' or 'cuda'
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.window = window
        self.center = center
        self.device = device
        
    def extract(self, audio: AudioData) -> SpectrogramData:
        """STFT抽出
        
        Args:
            audio: 音声データ
            
        Returns:
            SpectrogramData: 複素数、振幅(dB)、位相を含むスペクトログラム
        """
        # GPU転送
        waveform = self._to_device(audio.waveform)
        
        # STFT計算
        complex_spec = self._compute_stft(waveform)
        
        # 振幅・位相抽出
        magnitude = torch.abs(complex_spec)
        phase = torch.angle(complex_spec)
        
        # dBスケール変換
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
                'source_file': audio.metadata.get('file_path'),
                'device': self.device
            }
        )
    
    def extract_streaming(
        self, 
        audio_chunk: np.ndarray,
        state: Optional[Dict[str, Any]] = None
    ) -> tuple[SpectrogramData, Dict[str, Any]]:
        """ストリーミングSTFT抽出（Overlap-Add対応）
        
        Args:
            audio_chunk: 音声チャンク
            state: 前回の状態（バッファ等）
            
        Returns:
            SpectrogramData: 現在のチャンクのスペクトログラム
            Dict: 次回のための状態
        """
        pass
    
    def _compute_stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """STFT計算（torch.stft使用）"""
        pass
    
    def _to_db(
        self, 
        magnitude: torch.Tensor, 
        ref: float = 1.0,
        amin: float = 1e-10,
        top_db: float = 80.0
    ) -> torch.Tensor:
        """振幅をdBスケールに変換"""
        pass
    
    def _to_device(self, tensor: np.ndarray | torch.Tensor) -> torch.Tensor:
        """テンソルをデバイスに転送"""
        pass
```

### 4.2 ISTFT逆変換器（transforms/inverse.py）

```python
class ISTFTReconstructor:
    """ISTFT逆変換器
    
    振幅・位相スペクトログラムから音声波形を再構成。
    完全な可逆変換を実現。
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
    def reconstruct(self, spec: SpectrogramData) -> AudioData:
        """ISTFT逆変換
        
        Args:
            spec: スペクトログラムデータ（振幅・位相含む）
            
        Returns:
            AudioData: 再構成された音声波形
        """
        # 複素数スペクトログラムを使用（位相情報あり）
        complex_spec = spec.complex_spec
        
        # ISTFT計算
        waveform = torch.istft(
            complex_spec,
            n_fft=spec.n_fft,
            hop_length=spec.hop_length,
            win_length=spec.win_length,
            window=self._get_window(spec.window, spec.win_length),
            center=True
        )
        
        # クリッピング防止・正規化
        waveform = self._normalize(waveform)
        
        return AudioData(
            waveform=waveform.cpu().numpy(),
            sample_rate=spec.sample_rate,
            n_channels=1,
            duration=len(waveform) / spec.sample_rate,
            metadata={'reconstructed': True, 'method': 'istft'}
        )
    
    def reconstruct_streaming(
        self,
        spec_chunk: SpectrogramData,
        state: Optional[Dict[str, Any]] = None
    ) -> tuple[AudioData, Dict[str, Any]]:
        """ストリーミングISTFT（Overlap-Add）"""
        pass

class GriffinLimReconstructor:
    """Griffin-Limアルゴリズム逆変換器
    
    振幅スペクトログラムのみから位相を推定して音声を再構成。
    """
    
    def __init__(
        self,
        n_iter: int = 32,
        momentum: float = 0.99,
        device: str = 'cpu'
    ):
        """
        Args:
            n_iter: 反復回数
            momentum: モメンタム係数
            device: 計算デバイス
        """
        self.n_iter = n_iter
        self.momentum = momentum
        self.device = device
        
    def reconstruct(self, spec: SpectrogramData) -> AudioData:
        """Griffin-Lim逆変換
        
        Args:
            spec: スペクトログラムデータ（振幅のみ使用）
            
        Returns:
            AudioData: 再構成された音声波形
        """
        # dBスケールから線形スケールに変換
        magnitude = self._from_db(spec.magnitude_db)
        
        # 位相をランダム初期化
        phase = torch.rand_like(magnitude) * 2 * np.pi - np.pi
        
        # Griffin-Lim反復
        for _ in range(self.n_iter):
            # 複素数スペクトログラム生成
            complex_spec = magnitude * torch.exp(1j * phase)
            
            # ISTFT → STFT
            waveform = torch.istft(complex_spec, ...)
            new_complex_spec = torch.stft(waveform, ...)
            
            # 位相更新（モメンタム適用）
            new_phase = torch.angle(new_complex_spec)
            phase = self.momentum * phase + (1 - self.momentum) * new_phase
        
        # 最終的な波形生成
        final_complex_spec = magnitude * torch.exp(1j * phase)
        waveform = torch.istft(final_complex_spec, ...)
        
        return AudioData(
            waveform=waveform.cpu().numpy(),
            sample_rate=spec.sample_rate,
            n_channels=1,
            duration=len(waveform) / spec.sample_rate,
            metadata={'reconstructed': True, 'method': 'griffin-lim', 'n_iter': self.n_iter}
        )

### 4.3 Preprocessor（前処理・固定サイズ調整）（transforms/preprocessor.py）

```python
class AudioPreprocessor:
    """音声前処理器
    
    正規化、トリミング、固定長への調整（パディング・切り出し）を行う。
    バッチ処理で固定サイズの特徴量が必要な場合に使用。
    """
    
    def __init__(
        self,
        target_length: Optional[int] = None,
        sample_rate: int = 22050,
        normalize: bool = True,
        trim_silence: bool = False,
        trim_threshold_db: float = -40.0,
        padding_mode: str = 'constant',  # 'constant', 'reflect', 'replicate'
        device: str = 'cpu'
    ):
        """
        Args:
            target_length: 目標長（サンプル数）、Noneの場合は調整なし
            sample_rate: サンプリングレート
            normalize: 振幅正規化を行うか
            trim_silence: 無音区間除去を行うか
            trim_threshold_db: トリミング閾値（dB）
            padding_mode: パディングモード
            device: 計算デバイス
        """
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.trim_silence = trim_silence
        self.trim_threshold_db = trim_threshold_db
        self.padding_mode = padding_mode
        self.device = device
    
    def process(self, audio: AudioData) -> AudioData:
        """音声データを前処理
        
        Args:
            audio: 入力音声データ
            
        Returns:
            前処理済み音声データ
        """
        waveform = audio.waveform
        
        # PyTorchに変換（内部処理統一）
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).to(self.device)
        
        # 1. 無音区間除去
        if self.trim_silence:
            waveform = self._trim_silence(waveform)
        
        # 2. 正規化
        if self.normalize:
            waveform = self._normalize(waveform)
        
        # 3. 固定長への調整
        original_length = waveform.shape[-1]
        if self.target_length is not None:
            waveform = self._adjust_length(waveform, self.target_length)
        
        # メタデータ更新
        metadata = audio.metadata.copy()
        metadata.update({
            'preprocessed': True,
            'original_length': original_length,
            'target_length': self.target_length,
            'normalized': self.normalize,
            'trimmed': self.trim_silence
        })
        
        return AudioData(
            waveform=waveform,
            sample_rate=audio.sample_rate,
            n_channels=audio.n_channels,
            duration=waveform.shape[-1] / audio.sample_rate,
            metadata=metadata
        )
    
    def _normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        """振幅正規化（-1.0 ~ 1.0）"""
        max_val = torch.abs(waveform).max()
        if max_val > 0:
            waveform = waveform / max_val
        return waveform
    
    def _trim_silence(self, waveform: torch.Tensor) -> torch.Tensor:
        """無音区間除去
        
        エネルギーベースのトリミング
        """
        # エネルギー計算
        energy = waveform ** 2
        
        # 閾値計算（dB → リニア）
        threshold = 10 ** (self.trim_threshold_db / 20)
        
        # 閾値以上の区間を検出
        mask = energy > threshold
        
        if mask.any():
            # 最初と最後の非無音位置を検出
            indices = torch.nonzero(mask, as_tuple=False).squeeze()
            if indices.numel() > 0:
                start = indices[0].item()
                end = indices[-1].item() + 1
                waveform = waveform[..., start:end]
        
        return waveform
    
    def _adjust_length(
        self,
        waveform: torch.Tensor,
        target_length: int
    ) -> torch.Tensor:
        """固定長への調整（パディングまたは切り出し）
        
        Args:
            waveform: 入力波形 (... n_samples)
            target_length: 目標長
            
        Returns:
            調整済み波形 (... target_length)
        """
        current_length = waveform.shape[-1]
        
        if current_length == target_length:
            return waveform
        
        elif current_length < target_length:
            # パディング
            pad_length = target_length - current_length
            
            if self.padding_mode == 'constant':
                # ゼロパディング
                pad = torch.zeros(*waveform.shape[:-1], pad_length, device=waveform.device)
                waveform = torch.cat([waveform, pad], dim=-1)
            
            elif self.padding_mode == 'reflect':
                # 反射パディング
                waveform = torch.nn.functional.pad(
                    waveform,
                    (0, pad_length),
                    mode='reflect'
                )
            
            elif self.padding_mode == 'replicate':
                # 端の値を複製
                waveform = torch.nn.functional.pad(
                    waveform,
                    (0, pad_length),
                    mode='replicate'
                )
        
        else:
            # 切り出し（中央部分を使用）
            start = (current_length - target_length) // 2
            waveform = waveform[..., start:start + target_length]
        
        return waveform


class SpectrogramPreprocessor:
    """スペクトログラム前処理器
    
    時間軸方向の固定サイズ調整（パディング・切り出し）
    """
    
    def __init__(
        self,
        target_frames: Optional[int] = None,
        padding_value: float = -80.0,  # dBスケールの最小値
        crop_mode: str = 'center'  # 'center', 'random'
    ):
        """
        Args:
            target_frames: 目標フレーム数、Noneの場合は調整なし
            padding_value: パディング時の値（dBスケール）
            crop_mode: 切り出しモード（center: 中央、random: ランダム位置）
        """
        self.target_frames = target_frames
        self.padding_value = padding_value
        self.crop_mode = crop_mode
    
    def process(self, spec: SpectrogramData) -> SpectrogramData:
        """スペクトログラムを前処理
        
        Args:
            spec: 入力スペクトログラム
            
        Returns:
            前処理済みスペクトログラム
        """
        if self.target_frames is None:
            return spec
        
        original_frames = spec.magnitude_db.shape[0]
        
        # 調整
        magnitude_db = self._adjust_frames(spec.magnitude_db)
        phase = self._adjust_frames(spec.phase)
        complex_spec = self._adjust_frames(spec.complex_spec)
        
        # メタデータ更新
        metadata = spec.metadata.copy()
        metadata.update({
            'preprocessed': True,
            'original_frames': original_frames,
            'target_frames': self.target_frames
        })
        
        return SpectrogramData(
            complex_spec=complex_spec,
            magnitude_db=magnitude_db,
            phase=phase,
            n_fft=spec.n_fft,
            hop_length=spec.hop_length,
            win_length=spec.win_length,
            window=spec.window,
            sample_rate=spec.sample_rate,
            metadata=metadata
        )
    
    def _adjust_frames(self, data: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """フレーム数調整"""
        current_frames = data.shape[0]
        
        if current_frames == self.target_frames:
            return data
        
        elif current_frames < self.target_frames:
            # パディング
            pad_frames = self.target_frames - current_frames
            
            if isinstance(data, np.ndarray):
                pad_shape = (pad_frames,) + data.shape[1:]
                pad = np.full(pad_shape, self.padding_value, dtype=data.dtype)
                data = np.concatenate([data, pad], axis=0)
            else:  # torch.Tensor
                pad_shape = (pad_frames,) + data.shape[1:]
                pad = torch.full(pad_shape, self.padding_value, dtype=data.dtype, device=data.device)
                data = torch.cat([data, pad], dim=0)
        
        else:
            # 切り出し
            if self.crop_mode == 'center':
                start = (current_frames - self.target_frames) // 2
            elif self.crop_mode == 'random':
                import random
                start = random.randint(0, current_frames - self.target_frames)
            else:
                start = 0
            
            data = data[start:start + self.target_frames]
        
        return data
```

### 4.4 DatasetWriter（音声復元対応）（io/dataset_writer.py）

#### データ形式の種類

モデルの性能比較のため、4種類のデータ形式をサポート：

```python
from enum import Enum

class DatasetFormat(Enum):
    """データセット保存形式
    
    CNN入力を想定したチャンネル構成（time, freq, channels）
    """
    # 1. 複素数形式（実部・虚部）
    COMPLEX = "complex"  # channels=2: [real, imag]
    
    # 2. 振幅・位相形式（dBスケール・ラジアン）
    MAGNITUDE_PHASE = "magnitude_phase"  # channels=2: [mag_db, phase]
    
    # 3. 振幅・位相三角関数形式
    MAGNITUDE_PHASE_TRIG = "magnitude_phase_trig"  # channels=3: [mag_db, cos(phase), sin(phase)]
    
    # 4. 振幅のみ形式（Griffin-Lim復元用）
    MAGNITUDE_ONLY = "magnitude_only"  # channels=1: [mag_db]
```

**各形式の特徴**:

| 形式 | チャンネル | 用途 | 復元方法 | メリット | デメリット |
|------|-----------|------|---------|---------|----------|
| `COMPLEX` | 2ch | 位相学習（実部・虚部） | ISTFT | 完全復元可能 | 複素数の扱いが必要 |
| `MAGNITUDE_PHASE` | 2ch | 位相学習（極座標） | ISTFT | 直感的、完全復元可能 | 位相の周期性問題 |
| `MAGNITUDE_PHASE_TRIG` | 3ch | 位相学習（三角関数） | ISTFT | 連続性が高い | チャンネル数増 |
| `MAGNITUDE_ONLY` | 1ch | 振幅のみ学習 | Griffin-Lim | データ量少、単純 | 位相情報損失 |

**データ形状（CNN用）**:
```python
# HDF5保存形状: (time_frames, freq_bins, channels)
# 例: n_fft=2048, 3秒音声, sample_rate=22050
#   time_frames ≈ 129 (3秒 × 22050 / 512hop)
#   freq_bins = 1025 (n_fft // 2 + 1)

# COMPLEX形式
data.shape = (129, 1025, 2)  # [real, imag]

# MAGNITUDE_PHASE形式
data.shape = (129, 1025, 2)  # [mag_db, phase]

# MAGNITUDE_PHASE_TRIG形式
data.shape = (129, 1025, 3)  # [mag_db, cos, sin]

# MAGNITUDE_ONLY形式
data.shape = (129, 1025, 1)  # [mag_db]

# PyTorchでの使用時（CNNの一般的な形式）
data = torch.from_numpy(data).permute(2, 0, 1)  # (C, T, F)
# または batch次元含む
data = data.permute(0, 3, 1, 2)  # (B, C, T, F)
```

```python
class HDF5DatasetWriter:
    """HDF5形式データセット書き込み
    
    複数のデータ形式に対応し、モデル性能比較を可能にする。
    CNN入力を想定した (time, freq, channels) 形状で保存。
    """
    
    def __init__(
        self,
        format: DatasetFormat = DatasetFormat.MAGNITUDE_PHASE_TRIG,
        compression: str = 'gzip',
        compression_opts: int = 4
    ):
        """
        Args:
            format: データセット保存形式
            compression: 圧縮方式 ('gzip', 'lzf', None)
            compression_opts: 圧縮レベル (0-9、gzipの場合)
        """
        self.format = format
        self.compression = compression
        self.compression_opts = compression_opts
        self.converter = TensorConverter()
    
    def write(
        self,
        data: list[SpectrogramData],
        output_path: str,
        split: Optional[str] = None  # 'train', 'val', 'test'
    ) -> None:
        """データセットを書き込み
        
        Args:
            data: スペクトログラムデータリスト
            output_path: 出力HDF5ファイルパス
            split: データ分割名（train/val/test）
        """
        import h5py
        
        mode = 'a' if os.path.exists(output_path) else 'w'
        
        with h5py.File(output_path, mode) as f:
            # 分割グループ作成
            group_name = split if split else 'data'
            if group_name in f:
                group = f[group_name]
            else:
                group = f.create_group(group_name)
            
            for i, spec in enumerate(data):
                # PyTorch → NumPy
                spec_np = self.converter.ensure_numpy(spec)
                
                # サンプルグループ
                sample_key = f'sample_{len(group):06d}'  # 連番
                sample_group = group.create_group(sample_key)
                
                # === 特徴量データ（形式に応じて変換） ===
                channels_data = self._create_channels(spec_np)
                
                # CNN用データ保存: (time, freq, channels)
                sample_group.create_dataset(
                    'data',
                    data=channels_data,
                    compression=self.compression,
                    compression_opts=self.compression_opts
                )
                
                # === 音声復元用メタデータ ===
                # データ形式
                sample_group.attrs['format'] = self.format.value
                sample_group.attrs['n_channels'] = channels_data.shape[-1]
                
                # STFT パラメータ（ISTFT に必須）
                sample_group.attrs['n_fft'] = spec_np.n_fft
                sample_group.attrs['hop_length'] = spec_np.hop_length
                sample_group.attrs['win_length'] = spec_np.win_length
                sample_group.attrs['window'] = spec_np.window
                sample_group.attrs['sample_rate'] = spec_np.sample_rate
                
                # 元の音声情報
                if 'original_length' in spec_np.metadata:
                    sample_group.attrs['original_length'] = spec_np.metadata['original_length']
                
                if 'preprocessed' in spec_np.metadata:
                    sample_group.attrs['preprocessed'] = spec_np.metadata['preprocessed']
                    if spec_np.metadata.get('target_length'):
                        sample_group.attrs['target_length'] = spec_np.metadata['target_length']
                
                # ファイル情報
                if 'source_file' in spec_np.metadata:
                    sample_group.attrs['source_file'] = spec_np.metadata['source_file']
                
                # データ形状
                sample_group.attrs['n_frames'] = channels_data.shape[0]
                sample_group.attrs['n_freq_bins'] = channels_data.shape[1]
            
            # === グローバルメタデータ ===
            group.attrs['n_samples'] = len(data)
            group.attrs['format'] = self.format.value
            group.attrs['created_at'] = datetime.now().isoformat()
            
            # 共通パラメータ（全サンプルで同じ場合）
            if len(data) > 0:
                group.attrs['common_sample_rate'] = data[0].sample_rate
                group.attrs['common_n_fft'] = data[0].n_fft
                group.attrs['common_hop_length'] = data[0].hop_length
    
    def _create_channels(self, spec: SpectrogramData) -> np.ndarray:
        """データ形式に応じてチャンネルデータを生成
        
        Args:
            spec: スペクトログラムデータ
            
        Returns:
            channels_data: (time, freq, channels)
        """
        if self.format == DatasetFormat.COMPLEX:
            # [real, imag] - 2チャンネル
            channels = np.stack([
                np.real(spec.complex_spec),
                np.imag(spec.complex_spec)
            ], axis=-1)
        
        elif self.format == DatasetFormat.MAGNITUDE_PHASE:
            # [mag_db, phase] - 2チャンネル
            channels = np.stack([
                spec.magnitude_db,
                spec.phase
            ], axis=-1)
        
        elif self.format == DatasetFormat.MAGNITUDE_PHASE_TRIG:
            # [mag_db, cos(phase), sin(phase)] - 3チャンネル
            channels = np.stack([
                spec.magnitude_db,
                np.cos(spec.phase),
                np.sin(spec.phase)
            ], axis=-1)
        
        elif self.format == DatasetFormat.MAGNITUDE_ONLY:
            # [mag_db] - 1チャンネル
            channels = spec.magnitude_db[..., np.newaxis]
        
        else:
            raise ValueError(f"Unknown format: {self.format}")
        
        return channels


class AudioReconstructor:
    """モデル出力から音声を復元
    
    学習済みモデルの出力（予測されたスペクトログラム）から
    音声ファイルに戻す。
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.istft_reconstructor = ISTFTReconstructor(device=device)
        self.griffin_lim_reconstructor = GriffinLimReconstructor(device=device)
        self.converter = TensorConverter()
    
    def reconstruct_from_model_output(
        self,
        model_output: np.ndarray | torch.Tensor,
        format: DatasetFormat,
        stft_params: Dict[str, Any],
        method: str = 'auto',  # 'auto', 'istft', 'griffin-lim'
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
        format: DatasetFormat
    ) -> np.ndarray:
        """モデル出力から複素数スペクトログラムを復元
        
        Args:
            model_output: (time, freq, channels)
            format: データ形式
            
        Returns:
            complex_spec: (time, freq) - 複素数型
        """
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
    ) -> list[str]:
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
```

**データ形式比較表**:

| 形式 | HDF5保存内容 | 形状 | メリット | デメリット | 推奨用途 |
|------|------------|------|---------|----------|---------|
| `COMPLEX` | `[real, imag]` | (T, F, 2) | 完全復元、位相学習直接 | 複素数扱い必要 | 位相重視タスク |
| `MAGNITUDE_PHASE` | `[mag_db, phase]` | (T, F, 2) | 直感的、完全復元 | 位相不連続 | 汎用的 |
| `MAGNITUDE_PHASE_TRIG` | `[mag_db, cos, sin]` | (T, F, 3) | 位相連続、学習安定 | チャンネル数増 | 推奨（デフォルト） |
| `MAGNITUDE_ONLY` | `[mag_db]` | (T, F, 1) | 軽量、単純 | 位相情報なし | ノイズ除去等 |

**復元フロー例**:

```python
# === 学習フェーズ ===
# 1. データセット作成
writer = HDF5DatasetWriter(format=DatasetFormat.MAGNITUDE_PHASE_TRIG)
writer.write(spectrograms, 'train.h5', split='train')

# 2. PyTorchでデータロード
dataset = h5py.File('train.h5', 'r')
data = dataset['train']['sample_000000']['data'][:]  # (time, freq, 3)
data = torch.from_numpy(data).permute(2, 0, 1)  # (3, time, freq) for CNN

# 3. モデル学習
model.train()
output = model(data)  # 予測スペクトログラム

# === 推論・評価フェーズ ===
# 4. テストデータで推論
test_data = dataset['test']['sample_000000']['data'][:]
model_output = model(torch.from_numpy(test_data).permute(2, 0, 1))

# 5. 音声復元
reconstructor = AudioReconstructor(device='cuda')
audio = reconstructor.reconstruct_from_model_output(
    model_output=model_output.permute(1, 2, 0).cpu().numpy(),  # (time, freq, 3)
    format=DatasetFormat.MAGNITUDE_PHASE_TRIG,
    stft_params={'n_fft': 2048, 'hop_length': 512, ...},
    method='auto'  # 自動的にISTFT選択
)

# 6. 音声ファイル保存
sf.write('output.wav', audio.waveform, audio.sample_rate)
```
```

### 4.5 パイプライン実行器（pipeline/executor.py）

```python
class PipelineExecutor:
    """バッチ処理パイプライン実行器"""
    
    def __init__(
        self,
        loader: AudioLoader,
        extractors: list[FeatureExtractor],
        writer: DatasetWriter,
        device: str = 'cpu',
        batch_size: int = 16,
        num_workers: int = 4
    ):
        self.loader = loader
        self.extractors = extractors
        self.writer = writer
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.state_manager = StateManager()
        
    def execute(
        self,
        input_paths: list[str],
        output_path: str,
        progress_callback: Optional[Callable] = None
    ) -> ExecutionReport:
        """パイプライン実行
        
        Args:
            input_paths: 入力音声ファイルパス一覧
            output_path: 出力データセットパス
            progress_callback: 進捗コールバック
            
        Returns:
            ExecutionReport: 実行結果レポート
        """
        results = []
        
        # バッチ処理
        for batch_paths in self._create_batches(input_paths):
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
            
            if progress_callback:
                progress_callback(len(results), len(input_paths))
        
        return ExecutionReport(
            total_files=len(input_paths),
            successful=len(results),
            failed=len(input_paths) - len(results),
            processing_time=self.state_manager.get_elapsed_time()
        )

class StreamPipelineExecutor:
    """リアルタイム処理パイプライン実行器"""
    
    def __init__(
        self,
        extractors: list[FeatureExtractor],
        chunk_size: int = 1024,
        hop_size: int = 512,
        device: str = 'cpu'
    ):
        self.extractors = extractors
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.device = device
        self.buffer = StreamBuffer(chunk_size, hop_size)
        self.latency_monitor = LatencyMonitor()
        
    def process_chunk(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int
    ) -> list[SpectrogramData | FeatureData]:
        """音声チャンクを処理
        
        Args:
            audio_chunk: 音声データチャンク
            sample_rate: サンプリングレート
            
        Returns:
            抽出された特徴量リスト
        """
        start_time = time.time()
        
        # バッファに追加
        self.buffer.append(audio_chunk)
        
        # 十分なデータが溜まったら処理
        if self.buffer.is_ready():
            data = self.buffer.get_frame()
            
            # 特徴量抽出
            features = []
            for extractor in self.extractors:
                feature = extractor.extract_streaming(data, self.buffer.state)
                features.append(feature)
            
            # レイテンシ計測
            latency = time.time() - start_time
            self.latency_monitor.record(latency)
            
            return features
        
        return []

### 4.6 StreamBuffer詳細設計（pipeline/stream_buffer.py）

```python
class StreamBuffer:
    """リアルタイム処理用バッファ（Overlap-Add対応）
    
    音声ストリームをチャンク単位で受け取り、STFT処理に必要な
    フレームサイズ（n_fft）のデータを提供する。
    Overlap-Add法により連続的な処理を実現。
    """
    
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        device: str = 'cpu'
    ):
        """
        Args:
            n_fft: FFTフレームサイズ
            hop_length: ホップ長（フレーム間のオーバーラップ）
            device: 計算デバイス
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = device
        
        # 内部バッファ（n_fft分の音声データを保持）
        self.buffer = torch.zeros(n_fft, device=device)
        
        # バッファ内の有効データ数
        self.buffer_filled = 0
        
        # ISTFT用の重複部分保持（Overlap-Add）
        self.overlap_buffer = torch.zeros(n_fft - hop_length, device=device)
        
        # 統計情報
        self.total_chunks_received = 0
        self.total_frames_processed = 0
    
    def append(self, chunk: np.ndarray | torch.Tensor) -> None:
        """音声チャンクをバッファに追加
        
        Args:
            chunk: 音声データチャンク（1D array、任意の長さ）
        """
        # NumPy → PyTorch 変換（内部処理統一）
        if isinstance(chunk, np.ndarray):
            chunk = torch.from_numpy(chunk).to(self.device)
        
        chunk_size = len(chunk)
        
        # バッファの空き容量確認
        available_space = self.n_fft - self.buffer_filled
        
        if chunk_size <= available_space:
            # バッファに全て追加可能
            self.buffer[self.buffer_filled:self.buffer_filled + chunk_size] = chunk
            self.buffer_filled += chunk_size
        else:
            # バッファが溢れる場合、分割して処理
            # 今回の実装では簡略化：バッファサイズまで追加
            self.buffer[self.buffer_filled:] = chunk[:available_space]
            self.buffer_filled = self.n_fft
            
            # 残りのデータは次回処理（実装では警告を出す）
            if chunk_size > available_space:
                import warnings
                warnings.warn(f"Chunk size {chunk_size} exceeds buffer capacity. "
                            f"Only first {available_space} samples added.")
        
        self.total_chunks_received += 1
    
    def is_ready(self) -> bool:
        """処理可能なフレームがあるか判定
        
        Returns:
            True: n_fft分のデータが溜まっている
        """
        return self.buffer_filled >= self.n_fft
    
    def get_frame(self) -> torch.Tensor:
        """STFT処理用のフレームを取得
        
        Returns:
            n_fft サイズの音声フレーム
        """
        if not self.is_ready():
            raise RuntimeError("Buffer not ready. Check is_ready() before calling.")
        
        # フレーム取得
        frame = self.buffer.clone()
        
        # バッファをシフト（Overlap-Add）
        # hop_length分だけ進める
        self.buffer = torch.roll(self.buffer, -self.hop_length)
        
        # 新しく空いた部分をゼロクリア
        self.buffer[-self.hop_length:] = 0
        
        # 有効データ数を更新
        self.buffer_filled = max(0, self.buffer_filled - self.hop_length)
        
        self.total_frames_processed += 1
        
        return frame
    
    def get_overlap_buffer(self) -> torch.Tensor:
        """ISTFT用の重複部分バッファを取得"""
        return self.overlap_buffer.clone()
    
    def update_overlap(self, reconstructed_audio: torch.Tensor) -> torch.Tensor:
        """ISTFT結果とOverlap-Addして最終音声を生成
        
        Args:
            reconstructed_audio: ISTFT で再構成された音声（n_fft長）
            
        Returns:
            Overlap-Add済み音声（hop_length長）
        """
        # 重複部分を加算
        output_size = self.hop_length
        output = torch.zeros(output_size, device=self.device)
        
        # 前回の重複部分と加算
        overlap_size = min(len(self.overlap_buffer), output_size)
        output[:overlap_size] += self.overlap_buffer[:overlap_size]
        
        # 今回の出力の最初の部分も加算
        output += reconstructed_audio[:output_size]
        
        # 次回用の重複部分を保存
        self.overlap_buffer = reconstructed_audio[output_size:].clone()
        
        return output
    
    def reset(self) -> None:
        """バッファをリセット"""
        self.buffer.zero_()
        self.overlap_buffer.zero_()
        self.buffer_filled = 0
        self.total_chunks_received = 0
        self.total_frames_processed = 0
    
    @property
    def state(self) -> Dict[str, Any]:
        """現在の状態を取得（デバッグ・モニタリング用）"""
        return {
            'buffer_filled': self.buffer_filled,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'total_chunks_received': self.total_chunks_received,
            'total_frames_processed': self.total_frames_processed,
            'is_ready': self.is_ready()
        }


class LatencyMonitor:
    """レイテンシモニタリング"""
    
    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: 移動平均のウィンドウサイズ
        """
        self.window_size = window_size
        self.latencies = []
        
    def record(self, latency: float) -> None:
        """レイテンシを記録
        
        Args:
            latency: レイテンシ（秒）
        """
        self.latencies.append(latency)
        
        # ウィンドウサイズを超えたら古いデータを削除
        if len(self.latencies) > self.window_size:
            self.latencies.pop(0)
    
    def get_average(self) -> float:
        """平均レイテンシを取得（秒）"""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)
    
    def get_average_ms(self) -> float:
        """平均レイテンシを取得（ミリ秒）"""
        return self.get_average() * 1000
    
    def get_max(self) -> float:
        """最大レイテンシを取得（秒）"""
        return max(self.latencies) if self.latencies else 0.0
    
    def get_max_ms(self) -> float:
        """最大レイテンシを取得（ミリ秒）"""
        return self.get_max() * 1000
    
    def is_within_target(self, target_ms: float = 100.0) -> bool:
        """目標レイテンシ以内か判定
        
        Args:
            target_ms: 目標レイテンシ（ミリ秒）
        """
        return self.get_average_ms() <= target_ms
    
    def get_report(self) -> Dict[str, float]:
        """レイテンシレポートを取得"""
        return {
            'average_ms': self.get_average_ms(),
            'max_ms': self.get_max_ms(),
            'count': len(self.latencies),
            'within_100ms_target': self.is_within_target(100.0)
        }
```

**レイテンシ内訳計算**:

```python
# 想定パラメータ
sample_rate = 22050
chunk_size = 1024  # 音声チャンク
n_fft = 2048
hop_length = 512

# レイテンシ内訳
audio_chunk_duration = chunk_size / sample_rate  # ~46ms
stft_processing = 10-20  # ms (GPU使用時)
data_transfer = 5-10  # ms (CPU↔GPU)
overhead = 5  # ms (バッファ管理等)

# 合計: 46 + 20 + 10 + 5 = ~81ms
# 目標100ms以内 → 達成可能
```
```

## 5. データフロー

### 5.1 バッチ処理フロー

```
[音声ファイル群]
      ↓
 AudioLoader.load_batch()
      ↓
 [AudioData list] ──→ StateManager (stage: 'loaded')
      ↓
 STFTExtractor.extract()
      ↓
 [SpectrogramData list] ──→ StateManager (stage: 'extracted')
      ↓
 Preprocessor.transform()
      ↓
 [Transformed SpectrogramData] ──→ StateManager (stage: 'transformed')
      ↓
 DatasetWriter.write()
      ↓
 [Dataset files] ──→ StateManager (stage: 'saved')
```

### 5.2 リアルタイム処理フロー

```
[音声ストリーム]
      ↓
 StreamLoader.read_chunk() ──→ StreamBuffer
      ↓
 [AudioData chunk]
      ↓
 STFTExtractor.extract_streaming() (Overlap-Add)
      ↓
 [SpectrogramData chunk] ──→ LatencyMonitor
      ↓
 [出力/リアルタイム処理]
```

### 5.3 逆変換フロー

```
[SpectrogramData]
  (complex_spec or magnitude_db + phase)
      ↓
 ISTFTReconstructor.reconstruct()
  or
 GriffinLimReconstructor.reconstruct()
      ↓
 [AudioData (reconstructed)]
      ↓
 AudioWriter.save()
      ↓
 [音声ファイル]
```

## 6. 設定管理

### 6.1 設定スキーマ（config/schema.py）

```yaml
# config.yaml 例
audio:
  sample_rate: 22050
  n_channels: 1
  format: 'wav'

stft:
  n_fft: 2048
  hop_length: 512
  win_length: 2048
  window: 'hann'

features:
  - type: 'stft'
    output_channels: ['magnitude_db', 'phase']
  - type: 'mel_spectrogram'
    n_mels: 80
    fmin: 0
    fmax: 8000

preprocessing:
  normalize: true
  trim_silence: true
  threshold_db: -40

output:
  format: 'hdf5'
  compression: 'gzip'
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1

device:
  type: 'cuda'  # or 'cpu'
  gpu_id: 0

real_time:
  enabled: false
  chunk_size: 1024
  latency_target_ms: 100
```

### 6.2 設定Pythonクラス

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class STFTConfig:
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 2048
    window: Literal['hann', 'hamming', 'blackman'] = 'hann'

@dataclass
class FeatureConfig:
    type: Literal['stft', 'mel_spectrogram', 'mfcc']
    output_channels: list[str]
    params: dict

@dataclass
class PipelineConfig:
    audio: AudioConfig
    stft: STFTConfig
    features: list[FeatureConfig]
    preprocessing: PreprocessingConfig
    output: OutputConfig
    device: DeviceConfig
    real_time: RealTimeConfig
    
    @classmethod
    def from_yaml(cls, path: str) -> 'PipelineConfig':
        """YAMLファイルから設定を読み込み"""
        pass
```

## 7. GPU加速戦略

### 7.1 デバイス管理（core/device.py）

```python
class DeviceManager:
    """GPU/CPUデバイス管理"""
    
    def __init__(self, device: str = 'auto'):
        """
        Args:
            device: 'auto', 'cpu', 'cuda', 'cuda:0' etc.
        """
        self.device = self._resolve_device(device)
        
    def _resolve_device(self, device: str) -> torch.device:
        """デバイス解決"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """テンソルをデバイスに転送"""
        return tensor.to(self.device)
    
    def batch_to_device(self, batch: list[torch.Tensor]) -> list[torch.Tensor]:
        """バッチをデバイスに転送（並列化）"""
        pass
```

### 7.2 最適化ポイント

- **STFT/ISTFT**: `torch.stft` / `torch.istft` のGPU実装を使用
- **バッチ処理**: 複数ファイルを同時にGPUで処理
- **ストリーミング**: CUDAストリームを使った非同期処理
- **メモリ管理**: Pinned memoryを使った高速転送

## 8. テスト戦略

### 8.1 単体テスト

```python
# tests/test_stft_extractor.py
def test_stft_extraction():
    """STFT抽出のテスト"""
    # テスト用音声データ生成
    audio = generate_sine_wave(freq=440, duration=1.0, sr=22050)
    
    # STFT抽出
    extractor = STFTExtractor(n_fft=2048, hop_length=512)
    spec = extractor.extract(audio)
    
    # 検証
    assert spec.magnitude_db.shape[1] == 2048 // 2 + 1  # 周波数ビン数
    assert spec.phase.shape == spec.magnitude_db.shape
    assert spec.complex_spec.dtype == torch.complex64

def test_istft_reconstruction():
    """ISTFT逆変換のテスト"""
    # 元の音声
    original_audio = generate_sine_wave(freq=440, duration=1.0, sr=22050)
    
    # STFT → ISTFT
    extractor = STFTExtractor()
    spec = extractor.extract(original_audio)
    
    reconstructor = ISTFTReconstructor()
    reconstructed_audio = reconstructor.reconstruct(spec)
    
    # 高いSNRで再構成できることを確認
    snr = compute_snr(original_audio.waveform, reconstructed_audio.waveform)
    assert snr > 30.0  # 30dB以上
```

### 8.2 統合テスト

```python
# tests/test_pipeline.py
def test_end_to_end_pipeline():
    """End-to-Endパイプラインテスト"""
    # テストデータ準備
    test_audio_files = create_test_dataset(n_files=10)
    
    # パイプライン構築
    pipeline = PipelineBuilder()
        .with_loader(AudioFileLoader())
        .with_extractor(STFTExtractor())
        .with_writer(HDF5Writer())
        .build()
    
    # 実行
    report = pipeline.execute(test_audio_files, output_path='test_output.h5')
    
    # 検証
    assert report.successful == 10
    assert os.path.exists('test_output.h5')
    
    # データセット読み込み確認
    dataset = h5py.File('test_output.h5', 'r')
    assert 'magnitude_db' in dataset
    assert 'phase' in dataset
```

### 8.3 パフォーマンステスト

```python
def test_real_time_latency():
    """リアルタイム処理レイテンシテスト"""
    executor = StreamPipelineExecutor(
        extractors=[STFTExtractor()],
        chunk_size=1024,
        device='cuda'
    )
    
    # チャンク処理
    for chunk in generate_audio_stream():
        start = time.time()
        features = executor.process_chunk(chunk, sample_rate=22050)
        latency = time.time() - start
        
        # 100ms以下を確認
        assert latency < 0.1
```

## 9. エラーハンドリング

### 9.1 例外階層

```python
class DatasetGeneratorError(Exception):
    """基底例外クラス"""
    pass

class AudioLoadError(DatasetGeneratorError):
    """音声読み込みエラー"""
    pass

class FeatureExtractionError(DatasetGeneratorError):
    """特徴量抽出エラー"""
    pass

class InvalidConfigError(DatasetGeneratorError):
    """設定エラー"""
    pass

class DeviceError(DatasetGeneratorError):
    """デバイス関連エラー"""
    pass
```

### 9.2 エラー回復戦略

- **音声読み込み失敗**: スキップしてログに記録、処理継続
- **GPU OOM**: 自動的にCPUフォールバック
- **設定エラー**: 早期検出、詳細なエラーメッセージ表示

## 10. パフォーマンス最適化

### 10.1 最適化ポイント

1. **GPU活用**
   - バッチ処理でGPU使用率を最大化
   - CUDAストリームでオーバーラップ処理

2. **並列化**
   - マルチプロセス音声読み込み
   - DataLoaderパターン採用

3. **メモリ効率**
   - 大規模データセットは逐次処理
   - メモリマップドファイル使用

4. **キャッシング**
   - STFT結果の中間キャッシュ（オプション）

## 11. デプロイメント・パッケージング

### 11.1 依存関係（pyproject.toml）

```toml
[project]
name = "dataset-generator"
version = "0.1.0"
dependencies = [
    "torch>=2.0.0",
    "torchaudio>=2.0.0",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "librosa>=0.10.0",
    "soundfile>=0.12.0",
    "h5py>=3.8.0",
    "pyyaml>=6.0",
    "click>=8.1.0",
    "tqdm>=4.65.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.3.0",
    "pytest-cov>=4.0.0",
    "ruff>=0.0.260",
    "black>=23.3.0",
    "mypy>=1.2.0",
]
```

### 11.2 CLIエントリーポイント

```python
# cli/main.py
import click

@click.group()
def cli():
    """Dataset Generator CLI"""
    pass

@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--config', '-c', type=click.Path(exists=True))
@click.option('--device', '-d', default='auto')
def extract(input_dir, output_path, config, device):
    """音声特徴量抽出"""
    # パイプライン実行
    pass

@cli.command()
@click.argument('spec_file', type=click.Path(exists=True))
@click.argument('output_audio', type=click.Path())
@click.option('--method', type=click.Choice(['istft', 'griffin-lim']), default='istft')
def reconstruct(spec_file, output_audio, method):
    """音声再構成"""
    pass
```

---

**ステータス**: 設計フェーズ - 承認待ち  
**作成日**: 2026-01-19  
**次のステップ**: 設計を確認し、承認後に `/kiro-spec-tasks audio-feature-extraction` でタスク分解へ進んでください
