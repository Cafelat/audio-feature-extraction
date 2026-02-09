# 技術スタック

## 言語・ランタイム
- Python 3.13
- CUDA 12.8 (GPU加速)

## パッケージマネージャー
- uv (高速Pythonパッケージマネージャー)

## コア実装フレームワーク
- **PyTorch** (2.0+, CUDA対応) - テンソル計算・GPU加速の中核
- **NumPy/SciPy** - 数値計算、信号処理アルゴリズム
- **librosa/soundfile** - 音声ファイル読み込み・処理
- **torchaudio** - PyTorchベースの音声処理ユーティリティ
- **Pydantic** - データバリデーション（AudioData, SpectrogramData）
- **Click** - CLI フレームワーク
- **Rich** - TUI レンダリング

## 型安全性とプロトコル
- **Python 3.13 型ヒント**: すべての関数・メソッドに型アノテーション
- **Protocol** ベース: AudioLoader など拡張可能なインターフェース定義
- **Dataclass** 検証: __post_init__ による型チェック

## テスト・品質保証フレームワーク
- pytest (カバレッジ目標 80%+)
- pytest-cov - カバレッジ測定
- pytest-benchmark - パフォーマンス測定
- mypy - 静的型チェック (strict モード)
- ruff - リンター・フォーマッター
- black - コードフォーマッター

## 開発環境
- Dev Container (Ubuntu 22.04 + NVIDIA CUDA)
- Git, GitHub CLI
- Pre-configured: CUDA runtime, cuDNN shared libraries

## デバイス戦略
- torch.device で CPU/CUDA を動的に選択
- テンソル操作は device-agnostic（CPU/GPU 両対応）
- GPU が利用可能な場合は自動的に活用

## メモリ・パフォーマンス特性
- STFT/ISTFT: PyTorch の stft/istft 関数を活用
- リアルタイム処理: <100ms latency 目標
- バッチ処理: 可変サイズ入力パディング対応
- Griffin-Lim: 振幅のみからの音声復元（反復最適化）

## データ処理パターン
- **AudioMixer のメタデータ戦略**:
  - クリッピング対策後の実際のSN比を記録（`snr_db_actual`）
  - 正規化係数とクリッピングフラグを保持
  - ノイズ除去モデル評価時の振幅検証を可能にする
  - headroom = 0.01 （-40dB余裕）で正規化
- **デバイス管理**: `device` パラメータで CPU/GPU を動的選択
- **型変換**: TensorConverter で numpy ↔ torch 相互変換
- **検証戦略**: dataclass の `__post_init__` で shape/type チェック
