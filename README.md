# Audio Feature Extraction & Dataset Generator

音声データから機械学習用の特徴量を抽出し、データセットを生成するPythonライブラリ

## 概要

本プロジェクトは、音声認識・音声合成・音声分析タスク向けの前処理パイプラインを提供します。
リアルタイム処理、複数のデータ形式、完全な音声復元機能を備えています。

## 主要機能

- ✅ **4種類のデータ形式対応** - モデル性能比較が可能
  - COMPLEX: `[real, imag]`
  - MAGNITUDE_PHASE: `[mag_db, phase]`
  - MAGNITUDE_PHASE_TRIG: `[mag_db, cos, sin]` (推奨)
  - MAGNITUDE_ONLY: `[mag_db]`
- ✅ **リアルタイム処理対応** - 100ms以下のレイテンシ
- ✅ **STFT/ISTFT完全実装** - 位相情報を保持した可逆変換
- ✅ **Griffin-Lim逆変換** - 振幅のみからの音声復元
- ✅ **固定サイズ調整** - バッチ処理用パディング・切り出し
- ✅ **モデル出力からの音声復元** - 学習済みモデルの評価
- ✅ **GPU加速対応** - PyTorch/CUDA 12.8

## システム要件

- Python 3.13+
- CUDA 12.8 (GPU使用時)
- uv (パッケージ管理)

## インストール

\`\`\`bash
# uvでセットアップ
uv sync

# 開発用依存関係も含む
uv sync --all-extras
\`\`\`

## クイックスタート

\`\`\`python
from dataset_generator.io import AudioFileLoader
from dataset_generator.features import STFTExtractor
from dataset_generator.io import HDF5DatasetWriter, DatasetFormat

# 音声読み込み
loader = AudioFileLoader(device='cuda')
audio = loader.load('sample.wav')

# STFT抽出
extractor = STFTExtractor(n_fft=2048, hop_length=512, device='cuda')
spec = extractor.extract(audio)

# データセット保存（4種類から選択可能）
writer = HDF5DatasetWriter(format=DatasetFormat.MAGNITUDE_PHASE_TRIG)
writer.write([spec], 'dataset.h5', split='train')
\`\`\`

## データ形式比較

| 形式 | チャンネル | 形状 | 用途 | 復元方法 |
|------|----------|------|------|---------|
| COMPLEX | 2 | (T, F, 2) | 位相学習（直接） | ISTFT |
| MAGNITUDE_PHASE | 2 | (T, F, 2) | 位相学習（極座標） | ISTFT |
| MAGNITUDE_PHASE_TRIG | 3 | (T, F, 3) | 位相学習（三角関数）⭐ | ISTFT |
| MAGNITUDE_ONLY | 1 | (T, F, 1) | 振幅のみ | Griffin-Lim |

## プロジェクト状態

現在、仕様策定フェーズ（Phase 1）完了：

- ✅ 要件定義 (v1.1.0)
- ✅ 設計書 (v1.5.0)
- ✅ タスク分解 (26タスク、39日見積もり)
- ⏳ 実装 (未着手)

詳細は \`.kiro/specs/audio-feature-extraction/\` を参照

## 開発ワークフロー

本プロジェクトは **Kiro-style Spec-Driven Development** を採用しています。

\`\`\`bash
# 進捗確認
/kiro-spec-status audio-feature-extraction

# 実装開始（承認後）
/kiro-spec-impl audio-feature-extraction TASK-001,TASK-002
\`\`\`

## ライセンス

MIT License

## 関連ドキュメント

- [仕様書](.kiro/specs/audio-feature-extraction/requirements.md)
- [設計書](.kiro/specs/audio-feature-extraction/design.md)
- [タスク分解](.kiro/specs/audio-feature-extraction/tasks.md)
- [開発ガイド](AGENTS.md)
