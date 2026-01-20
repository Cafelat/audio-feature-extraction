# タスク分解: 音声特徴量抽出モジュール及びデータセットジェネレータ

## タスク概要

設計書に基づき、実装可能な具体的タスクに分解。依存関係を考慮した実装順序で整理。

## タスク一覧サマリー

| フェーズ | タスク数 | 総見積もり工数 |
|---------|---------|--------------|
| **Phase 0: セットアップ** | 1 | 0.5日 |
| **Phase 1: P0（必須）** | 10 | 16.0日 |
| **Phase 2: P1（高優先度）** | 8 | 13.0日 |
| **Phase 3: P2（中優先度）** | 5 | 7.0日 |
| **Phase 4: P3（低優先度）** | 3 | 4.0日 |
| **合計** | 27 | 40.5日 |

---

## Phase 0: プロジェクトセットアップ

### TASK-000: プロジェクト環境構築
- **優先度**: P0（必須）
- **依存**: なし
- **見積もり**: 0.5日

#### 実装内容
- ディレクトリ構造作成（設計書2.1に準拠）
- `pyproject.toml` 作成（依存関係定義）
- `uv` による依存関係インストール
- 基本的な `__init__.py` ファイル配置
- `.gitignore`, `README.md` 初期版作成

#### 完了条件
- [ ] `src/dataset_generator/` 以下のディレクトリ構造が設計書通り作成されている
- [ ] PyTorch, librosa, soundfile, h5py等の依存関係がインストールされている
- [ ] `uv sync` でプロジェクトが正常にセットアップできる

#### ファイル
```
pyproject.toml
src/dataset_generator/__init__.py
src/dataset_generator/core/__init__.py
src/dataset_generator/io/__init__.py
src/dataset_generator/features/__init__.py
src/dataset_generator/transforms/__init__.py
src/dataset_generator/pipeline/__init__.py
src/dataset_generator/config/__init__.py
src/dataset_generator/cli/__init__.py
src/dataset_generator/utils/__init__.py
```

---

## Phase 1: P0（必須）- 基本機能実装

### TASK-001: コアデータモデル実装
- **優先度**: P0（必須）
- **依存**: TASK-000
- **見積もり**: 2.0日

#### 実装内容
- `AudioData` データクラス実装（設計書 3.1）
  - `__post_init__()` によるデータ検証（形状、型、整合性）
  - サンプル数・継続時間の検証
- `SpectrogramData` データクラス実装
  - 複素数スペクトログラム、振幅（dB）、位相の保持
  - `__post_init__()` による形状・型検証
  - `to_channels()` メソッド実装
  - `save_params()` メソッド実装
- `MelSpectrogramData` データクラス実装
- `FeatureData` データクラス実装
- `ProcessingState` データクラス実装

#### 完了条件
- [ ] 全データクラスが設計書通りに実装されている
- [ ] `__post_init__()` でデータ検証が正常に動作する
- [ ] 不正なデータで適切な例外が発生する
- [ ] 単体テストで80%以上のカバレッジ
- [ ] Type hints完備、mypy チェック通過

#### ファイル
```
src/dataset_generator/core/models.py
tests/core/test_models.py
```

---

### TASK-002: Protocol インターフェース定義
- **優先度**: P0（必須）
- **依存**: TASK-001
- **見積もり**: 0.5日

#### 実装内容
- `AudioLoader` Protocol 定義（設計書 3.2）
- `FeatureExtractor` Protocol 定義
- `InverseTransform` Protocol 定義
- `DatasetWriter` Protocol 定義
- `@runtime_checkable` デコレータ適用

#### 完了条件
- [ ] 全Protocolが設計書通りに定義されている
- [ ] `isinstance()` による実行時チェックが可能
- [ ] Type hints完備

#### ファイル
```
src/dataset_generator/core/types.py
tests/core/test_types.py
```

---

### TASK-003: TensorConverter実装
- **優先度**: P0（必須）
- **依存**: TASK-001
- **見積もり**: 1.5日

#### 実装内容
- `TensorConverter` クラス実装（設計書 3.3）
  - `to_torch()`: NumPy → PyTorch 変換（デバイス指定対応）
  - `to_numpy()`: PyTorch → NumPy 変換（GPU→CPU自動転送）
  - `ensure_torch()`: データクラス内の全配列をPyTorchに変換
  - `ensure_numpy()`: データクラス内の全TensorをNumPyに変換
- 複素数データ型の適切な処理
- 勾配追跡の解除処理

#### 完了条件
- [ ] NumPy ↔ PyTorch 双方向変換が正常に動作する
- [ ] GPU/CPU間の転送が正常に動作する
- [ ] `AudioData`, `SpectrogramData` に対する変換が動作する
- [ ] 単体テストで80%以上のカバレッジ

#### ファイル
```
src/dataset_generator/core/conversions.py
tests/core/test_conversions.py
```

---

### TASK-004: 音声ファイル読み込み実装
- **優先度**: P0（必須）
- **依存**: TASK-001, TASK-003
- **見積もり**: 1.5日

#### 実装内容
- `AudioFileLoader` クラス実装（設計書 3.3 使用例）
  - WAV, FLAC, MP3, OGG対応（soundfile/librosa使用）
  - `load()`: 単一ファイル読み込み（NumPy → PyTorch変換）
  - `load_batch()`: 複数ファイル一括読み込み
  - サンプリングレート変換（リサンプリング）
  - モノラル/ステレオ対応、チャンネル選択
- エラーハンドリング（`AudioLoadError`）

#### 完了条件
- [ ] WAV/FLACファイルの読み込みが動作する
- [ ] リサンプリングが正常に動作する
- [ ] モノラル/ステレオの処理が正常に動作する
- [ ] バッチ読み込みが動作する
- [ ] 単体テストで80%以上のカバレッジ

#### ファイル
```
src/dataset_generator/io/audio_loader.py
tests/io/test_audio_loader.py
```

---

### TASK-005: STFT抽出器実装
- **優先度**: P0（必須）
- **依存**: TASK-001, TASK-003, TASK-004
- **見積もり**: 2.5日

#### 実装内容
- `STFTExtractor` クラス実装（設計書 4.1）
  - `extract()`: STFT抽出（振幅・位相・複素数スペクトログラム）
  - `_compute_stft()`: `torch.stft()` を使用したSTFT計算
  - `_to_db()`: 振幅をdBスケールに変換（20*log10、クランピング）
  - `_to_device()`: デバイス転送
  - パラメータ: n_fft, hop_length, win_length, window
  - GPU加速対応
- 複素数スペクトログラムの正しい生成（`return_complex=True`）

#### 完了条件
- [ ] STFT抽出が正常に動作する
- [ ] 振幅（dB）、位相、複素数が正しく計算されている
- [ ] 周波数ビン数が `n_fft // 2 + 1` と一致する
- [ ] GPU/CPU両方で動作する
- [ ] 単体テストで80%以上のカバレッジ
- [ ] 正弦波テストでピークが正しい位置に出現する

#### ファイル
```
src/dataset_generator/features/stft.py
tests/features/test_stft.py
```

---

### TASK-006: ISTFT逆変換器実装
- **優先度**: P0（必須）
- **依存**: TASK-001, TASK-003, TASK-005
- **見積もり**: 2.0日

#### 実装内容
- `ISTFTReconstructor` クラス実装（設計書 4.2）
  - `reconstruct()`: ISTFT逆変換（複素数スペクトログラムから音声復元）
  - `torch.istft()` 使用
  - `_get_window()`: 窓関数生成
  - `_normalize()`: クリッピング防止、正規化
  - GPU加速対応
- 位相情報を利用した完全可逆変換

#### 完了条件
- [ ] ISTFT逆変換が正常に動作する
- [ ] STFT → ISTFT で元の音声が高精度で復元される（SNR > 30dB）
- [ ] GPU/CPU両方で動作する
- [ ] 単体テストで80%以上のカバレッジ

#### ファイル
```
src/dataset_generator/transforms/inverse.py
tests/transforms/test_inverse.py
```

---

### TASK-007: HDF5データセット書き込み実装
- **優先度**: P0（必須）
- **依存**: TASK-001, TASK-003
- **見積もり**: 2.5日

#### 実装内容
- `HDF5DatasetWriter` クラス実装（設計書 4.4）
  - `write()`: HDF5形式でデータセット書き込み
  - **4種類のデータ形式対応**:
    - `COMPLEX`: `[real, imag]` (2ch)
    - `MAGNITUDE_PHASE`: `[mag_db, phase]` (2ch)
    - `MAGNITUDE_PHASE_TRIG`: `[mag_db, cos, sin]` (3ch) - デフォルト
    - `MAGNITUDE_ONLY`: `[mag_db]` (1ch)
  - `_create_channels()`: 形式に応じたチャンネルデータ生成
  - CNN入力用の形状: `(time, freq, channels)`
  - 音声復元用メタデータ保存
    - データ形式（format）
    - ISTFT必須パラメータ（n_fft, hop_length, win_length, window, sample_rate）
    - 元の音声長（original_length）
  - 圧縮対応（gzip level 4デフォルト）
  - データ分割対応（train/val/test）

#### 完了条件
- [ ] HDF5ファイルへの書き込みが動作する
- [ ] 4種類のデータ形式すべてで保存できる
- [ ] 各形式のチャンネル数が正しい（2/2/3/1）
- [ ] データ形状が (time, freq, channels) になっている
- [ ] 全メタデータが保存される
- [ ] 圧縮が動作する
- [ ] 単体テストで80%以上のカバレッジ

#### ファイル
```
src/dataset_generator/io/dataset_writer.py
tests/io/test_dataset_writer.py
```

---

### TASK-008: 音声復元器実装
- **優先度**: P0（必須）
- **依存**: TASK-006, TASK-007
- **見積もり**: 2.0日

#### 実装内容
- `AudioReconstructor` クラス実装（設計書 4.4）
  - **`reconstruct_from_model_output()`**: モデル出力から音声復元（主要機能）
    - 4種類のデータ形式に対応
    - `_reconstruct_complex_spec()`: 形式に応じて複素数スペクトログラム復元
      - COMPLEX: `[real, imag]` → complex
      - MAGNITUDE_PHASE: `[mag_db, phase]` → magnitude * exp(1j*phase)
      - MAGNITUDE_PHASE_TRIG: `[mag_db, cos, sin]` → magnitude * exp(1j*arctan2(sin, cos))
      - MAGNITUDE_ONLY: `[mag_db]` → magnitude (位相なし、Griffin-Lim用)
    - 復元方法の自動選択（'auto'）
    - バッチ次元の処理
  - `reconstruct_from_dataset()`: データセットから音声復元（検証用）
  - `batch_reconstruct_from_dataset()`: データセット全体の一括復元
  - 元の長さへの調整（パディング除去）
  - soundfile による音声ファイル出力

#### 完了条件
- [ ] モデル出力（4種類すべての形式）から音声が正常に復元される
- [ ] 各形式で複素数スペクトログラムが正しく復元される
- [ ] データセットからの復元が動作する
- [ ] パディングが正しく除去される
- [ ] バッチ復元が動作する
- [ ] 音声ファイルとして保存できる
- [ ] 単体テストで80%以上のカバレッジ

#### ファイル
```
src/dataset_generator/io/audio_reconstructor.py
tests/io/test_audio_reconstructor.py
```

---

### TASK-009: 基本パイプライン実装
- **優先度**: P0（必須）
- **依存**: TASK-004, TASK-005, TASK-007
- **見積もり**: 1.5日

#### 実装内容
- `PipelineExecutor` クラス実装（設計書 4.5）
  - `execute()`: バッチ処理パイプライン実行
  - ローダー、抽出器、ライターの統合
  - バッチ処理対応
  - 進捗コールバック対応
  - `ExecutionReport` 生成
- `StateManager` クラス実装（簡易版）

#### 完了条件
- [ ] End-to-Endパイプラインが動作する
- [ ] 複数ファイルのバッチ処理が動作する
- [ ] 実行レポートが正しく生成される
- [ ] 統合テストで動作確認

#### ファイル
```
src/dataset_generator/pipeline/executor.py
src/dataset_generator/pipeline/state_manager.py
tests/pipeline/test_executor.py
```

---

### TASK-010: 基本CLI実装
- **優先度**: P0（必須）
- **依存**: TASK-009, TASK-008
- **見積もり**: 1.0日

#### 実装内容
- `cli/main.py` 実装（設計書 11.2）
  - `extract` コマンド: 音声特徴量抽出
  - `reconstruct` コマンド: 音声再構成
  - Click による CLI 構築
  - 基本的なオプション（--device, --config）
  - プログレスバー表示（tqdm）

#### 完了条件
- [ ] `extract` コマンドで特徴量抽出が動作する
- [ ] `reconstruct` コマンドで音声復元が動作する
- [ ] ヘルプメッセージが表示される
- [ ] 実際の音声ファイルでEnd-to-Endテスト成功

#### ファイル
```
src/dataset_generator/cli/main.py
tests/cli/test_main.py
```

---

## Phase 2: P1（高優先度）- 拡張機能

### TASK-011-EX: AudioMixer実装（SN比ノイズ重畳）
- **優先度**: P1（高優先度）
- **依存**: TASK-001, TASK-003
- **見積もり**: 1.5日

#### 実装内容
- `AudioMixer` クラス実装（設計書 4.3）
  - `mix_with_snr()`: 指定したSN比でクリーン信号とノイズを混合
    - RMS計算（Root Mean Square）
    - SNRに基づくノイズゲイン計算
    - ノイズ長の調整（パディング・切り出し・繰り返し）
  - `_adjust_noise_length()`: ノイズの長さをクリーン信号に合わせる
  - `_calculate_rms()`: RMS計算
  - `batch_mix()`: バッチ処理（ランダムなノイズ・SNR選択）
  - サンプリングレートチェック
  - モノラル化処理

#### 完了条件
- [ ] 指定したSN比でノイズ混合が動作する
- [ ] RMS計算が正しく実装されている
- [ ] ノイズの長さ調整が動作する（短い・長い両方）
- [ ] バッチ処理が動作する
- [ ] メタデータに混合情報が保存される
- [ ] 単体テストで80%以上のカバレッジ
- [ ] SNRの検証テスト（実測値が目標値±1dB以内）

#### ファイル
```
src/dataset_generator/transforms/preprocessor.py (拡張)
tests/transforms/test_audio_mixer.py
```

---

### TASK-011: AudioPreprocessor実装
- **優先度**: P1（高優先度）
- **依存**: TASK-001, TASK-003
- **見積もり**: 2.0日

#### 実装内容
- `AudioPreprocessor` クラス実装（設計書 4.3）
  - `process()`: 音声前処理
  - `_normalize()`: 振幅正規化
  - `_trim_silence()`: 無音区間除去（エネルギーベース）
  - `_adjust_length()`: 固定長への調整
    - パディング: constant/reflect/replicate
    - 切り出し: 中央切り出し
  - 元の長さをメタデータに保存

#### 完了条件
- [ ] 正規化が正常に動作する
- [ ] 無音区間除去が動作する
- [ ] 固定長調整（パディング・切り出し）が動作する
- [ ] メタデータが正しく保存される
- [ ] 単体テストで80%以上のカバレッジ

#### ファイル
```
src/dataset_generator/transforms/preprocessor.py
tests/transforms/test_preprocessor.py
```

---

### TASK-012: SpectrogramPreprocessor実装
- **優先度**: P1（高優先度）
- **依存**: TASK-001, TASK-005
- **見積もり**: 1.5日

#### 実装内容
- `SpectrogramPreprocessor` クラス実装（設計書 4.3）
  - `process()`: スペクトログラム前処理
  - `_adjust_frames()`: フレーム数調整
    - パディング: -80dBで埋める
    - 切り出し: center/random モード
  - メタデータ更新

#### 完了条件
- [ ] フレーム数調整が動作する
- [ ] パディング値が正しく設定される
- [ ] center/random切り出しが動作する
- [ ] 単体テストで80%以上のカバレッジ

#### ファイル
```
src/dataset_generator/transforms/preprocessor.py (拡張)
tests/transforms/test_spectrogram_preprocessor.py
```

---

### TASK-013: Griffin-Lim逆変換実装
- **優先度**: P1（高優先度）
- **依存**: TASK-006
- **見積もり**: 2.0日

#### 実装内容
- `GriffinLimReconstructor` クラス実装（設計書 4.2）
  - `reconstruct()`: Griffin-Limアルゴリズム実装
  - `_from_db()`: dBスケールから線形スケールへ変換
  - 反復的な位相推定（デフォルト32回）
  - モメンタム適用（デフォルト0.99）
  - GPU加速対応

#### 完了条件
- [ ] Griffin-Lim逆変換が動作する
- [ ] 振幅スペクトログラムのみから音声が生成される
- [ ] 反復回数による品質向上が確認できる
- [ ] GPU/CPU両方で動作する
- [ ] 単体テストで80%以上のカバレッジ

#### ファイル
```
src/dataset_generator/transforms/inverse.py (拡張)
tests/transforms/test_griffin_lim.py
```

---

### TASK-014: StreamBuffer実装
- **優先度**: P1（高優先度）
- **依存**: TASK-001, TASK-003
- **見積もり**: 2.5日

#### 実装内容
- `StreamBuffer` クラス実装（設計書 4.6）
  - `append()`: 音声チャンクをバッファに追加
  - `is_ready()`: 処理可能か判定
  - `get_frame()`: STFTフレーム取得、バッファシフト
  - `get_overlap_buffer()`: ISTFT用重複バッファ
  - `update_overlap()`: Overlap-Add処理
  - `reset()`: バッファリセット
  - `state` プロパティ: デバッグ用状態取得
- `LatencyMonitor` クラス実装
  - `record()`: レイテンシ記録
  - `get_average_ms()`: 平均レイテンシ取得
  - `is_within_target()`: 目標レイテンシ判定
  - `get_report()`: レポート生成

#### 完了条件
- [ ] バッファリングが正常に動作する
- [ ] Overlap-Add処理が正しく実装されている
- [ ] レイテンシモニタリングが動作する
- [ ] 単体テストで80%以上のカバレッジ

#### ファイル
```
src/dataset_generator/pipeline/stream_buffer.py
tests/pipeline/test_stream_buffer.py
```

---

### TASK-015: ストリーミング処理実装
- **優先度**: P1（高優先度）
- **依存**: TASK-005, TASK-006, TASK-014
- **見積もり**: 2.5日

#### 実装内容
- `STFTExtractor.extract_streaming()` 実装（設計書 4.1）
- `ISTFTReconstructor.reconstruct_streaming()` 実装（設計書 4.2）
- `StreamPipelineExecutor` クラス実装（設計書 4.5）
  - `process_chunk()`: 音声チャンク処理
  - StreamBuffer統合
  - LatencyMonitor統合
- リアルタイム処理フロー実装

#### 完了条件
- [ ] チャンクベースのSTFT抽出が動作する
- [ ] ストリーミングISTFTが動作する
- [ ] レイテンシが100ms以内に収まる（GPU使用時）
- [ ] 統合テストで連続処理が動作する

#### ファイル
```
src/dataset_generator/features/stft.py (拡張)
src/dataset_generator/transforms/inverse.py (拡張)
src/dataset_generator/pipeline/stream_executor.py
tests/pipeline/test_stream_executor.py
```

---

### TASK-016: YAML設定管理実装
- **優先度**: P1（高優先度）
- **依存**: TASK-001
- **見積もり**: 1.5日

#### 実装内容
- 設定データクラス実装（設計書 6.2）
  - `STFTConfig`
  - `FeatureConfig`
  - `PreprocessingConfig`
  - `OutputConfig`
  - `DeviceConfig`
  - `RealTimeConfig`
  - `PipelineConfig`
- `PipelineConfig.from_yaml()` 実装
- YAML設定ファイル読み込み・検証

#### 完了条件
- [ ] YAML設定ファイルが正しく読み込まれる
- [ ] 設定検証（バリデーション）が動作する
- [ ] 不正な設定で適切なエラーが発生する
- [ ] 単体テストで80%以上のカバレッジ

#### ファイル
```
src/dataset_generator/config/schema.py
src/dataset_generator/config/loader.py
tests/config/test_schema.py
tests/config/test_loader.py
```

---

### TASK-017: GPU加速・デバイス管理実装
- **優先度**: P1（高優先度）
- **依存**: TASK-003
- **見積もり**: 1.0日

#### 実装内容
- `DeviceManager` クラス実装（設計書 7.1）
  - `_resolve_device()`: デバイス自動解決（auto/cpu/cuda）
  - `to_device()`: テンソル転送
  - `batch_to_device()`: バッチ転送
- GPU OOM 時のCPUフォールバック処理
- CUDA利用可否の判定

#### 完了条件
- [ ] デバイス自動解決が動作する
- [ ] GPU/CPU切り替えが正常に動作する
- [ ] GPU未搭載環境でCPUフォールバックが動作する
- [ ] 単体テストで80%以上のカバレッジ

#### ファイル
```
src/dataset_generator/core/device.py
tests/core/test_device.py
```

---

## Phase 3: P2（中優先度）- 追加特徴量

### TASK-018: メルスペクトログラム抽出実装
- **優先度**: P2（中優先度）
- **依存**: TASK-005
- **見積もり**: 2.0日

#### 実装内容
- `MelSpectrogramExtractor` クラス実装
  - `extract()`: メルスペクトログラム抽出
  - librosa または torchaudio のメルフィルタバンク使用
  - dBスケール変換
  - パラメータ: n_mels, fmin, fmax

#### 完了条件
- [ ] メルスペクトログラム抽出が動作する
- [ ] dBスケール変換が正しく実装されている
- [ ] GPU/CPU両方で動作する
- [ ] 単体テストで80%以上のカバレッジ

#### ファイル
```
src/dataset_generator/features/mel.py
tests/features/test_mel.py
```

---

### TASK-019: MFCC抽出実装
- **優先度**: P2（中優先度）
- **依存**: TASK-018
- **見積もり**: 1.5日

#### 実装内容
- `MFCCExtractor` クラス実装
  - `extract()`: MFCC抽出
  - librosa 使用
  - デルタ・デルタデルタ算出オプション
  - パラメータ: n_mfcc

#### 完了条件
- [ ] MFCC抽出が動作する
- [ ] デルタ特徴量が算出できる
- [ ] 単体テストで80%以上のカバレッジ

#### ファイル
```
src/dataset_generator/features/mfcc.py
tests/features/test_mfcc.py
```

---

### TASK-020: データ拡張実装
- **優先度**: P2（中優先度）
- **依存**: TASK-001, TASK-003
- **見積もり**: 2.5日

#### 実装内容
- `DataAugmentation` クラス実装
  - ピッチシフト（librosa使用）
  - タイムストレッチ（librosa使用）
  - ノイズ付加（ガウシアンノイズ）
  - ランダム適用機能

#### 完了条件
- [ ] ピッチシフトが動作する
- [ ] タイムストレッチが動作する
- [ ] ノイズ付加が動作する
- [ ] 単体テストで80%以上のカバレッジ

#### ファイル
```
src/dataset_generator/transforms/augmentation.py
tests/transforms/test_augmentation.py
```

---

### TASK-021: データセット分割実装
- **優先度**: P2（中優先度）
- **依存**: TASK-007
- **見積もり**: 1.0日

#### 実装内容
- `DatasetSplitter` クラス実装
  - `split()`: train/val/test分割
  - 比率指定対応
  - ランダムシード固定

#### 完了条件
- [ ] データセット分割が動作する
- [ ] 指定比率通りに分割される
- [ ] シード固定で再現可能
- [ ] 単体テストで80%以上のカバレッジ

#### ファイル
```
src/dataset_generator/io/dataset_splitter.py
tests/io/test_dataset_splitter.py
```

---

### TASK-022: その他音響特徴量実装
- **優先度**: P2（中優先度）
- **依存**: TASK-005
- **見積もり**: 2.0日

#### 実装内容
- `ChromaExtractor` クラス実装
- `SpectralCentroidExtractor` クラス実装
- `ZeroCrossingRateExtractor` クラス実装
- librosa使用

#### 完了条件
- [ ] Chroma特徴量が抽出できる
- [ ] Spectral Centroidが抽出できる
- [ ] Zero Crossing Rateが抽出できる
- [ ] 単体テストで80%以上のカバレッジ

#### ファイル
```
src/dataset_generator/features/chroma.py
src/dataset_generator/features/spectral.py
tests/features/test_additional_features.py
```

---

## Phase 4: P3（低優先度）- 最適化・拡張

### TASK-023: カスタム特徴量抽出器レジストリ実装
- **優先度**: P3（低優先度）
- **依存**: TASK-002, TASK-005
- **見積もり**: 1.5日

#### 実装内容
- `FeatureExtractorRegistry` クラス実装
  - `register()`: カスタム抽出器登録
  - `get()`: 抽出器取得
  - Protocol準拠チェック

#### 完了条件
- [ ] カスタム抽出器の登録が動作する
- [ ] 登録した抽出器が使用できる
- [ ] 単体テストで80%以上のカバレッジ

#### ファイル
```
src/dataset_generator/features/registry.py
tests/features/test_registry.py
```

---

### TASK-024: 増分データセット更新実装
- **優先度**: P3（低優先度）
- **依存**: TASK-007
- **見積もり**: 1.5日

#### 実装内容
- `HDF5DatasetWriter` 拡張
  - 既存データセットへの追加機能
  - サンプル番号の自動インクリメント
  - メタデータの更新

#### 完了条件
- [ ] 既存データセットへの追加が動作する
- [ ] サンプル番号が正しく割り当てられる
- [ ] 単体テストで80%以上のカバレッジ

#### ファイル
```
src/dataset_generator/io/dataset_writer.py (拡張)
tests/io/test_incremental_update.py
```

---

### TASK-025: チェックポイント・エラー回復実装
- **優先度**: P3（低優先度）
- **依存**: TASK-009
- **見積もり**: 2.0日

#### 実装内容
- `CheckpointManager` クラス実装
  - 処理中断時の状態保存
  - チェックポイントからの再開機能
  - 処理済みファイルのトラッキング

#### 完了条件
- [ ] チェックポイント保存が動作する
- [ ] 再開処理が動作する
- [ ] 処理済みファイルがスキップされる
- [ ] 単体テストで80%以上のカバレッジ

#### ファイル
```
src/dataset_generator/pipeline/checkpoint.py
tests/pipeline/test_checkpoint.py
```

---

## タスク依存関係グラフ

```
TASK-000: プロジェクトセットアップ
    │
    ├─→ TASK-001: コアデータモデル
    │       ├─→ TASK-002: Protocol定義
    │       ├─→ TASK-003: TensorConverter
    │       │       ├─→ TASK-004: 音声読み込み
    │       │       │       └─→ TASK-009: 基本パイプライン ─→ TASK-010: CLI
    │       │       ├─→ TASK-005: STFT抽出
    │       │       │       ├─→ TASK-006: ISTFT逆変換
    │       │       │       │       ├─→ TASK-008: 音声復元 ─→ TASK-010
    │       │       │       │       ├─→ TASK-013: Griffin-Lim
    │       │       │       │       └─→ TASK-015: ストリーミング処理
    │       │       │       ├─→ TASK-012: SpectrogramPreprocessor
    │       │       │       ├─→ TASK-018: メルスペクトログラム
    │       │       │       │       └─→ TASK-019: MFCC
    │       │       │       └─→ TASK-022: その他音響特徴量
    │       │       ├─→ TASK-007: HDF5書き込み
    │       │       │       ├─→ TASK-008
    │       │       │       ├─→ TASK-021: データセット分割
    │       │       │       └─→ TASK-024: 増分更新
    │       │       ├─→ TASK-011: AudioPreprocessor
    │       │       ├─→ TASK-014: StreamBuffer
    │       │       │       └─→ TASK-015
    │       │       ├─→ TASK-017: GPU加速
    │       │       └─→ TASK-020: データ拡張
    │       └─→ TASK-016: YAML設定管理
    │
    ├─→ TASK-023: カスタム抽出器レジストリ
    └─→ TASK-025: チェックポイント
```

---

## 実装フェーズ推奨順序

### Week 1-2: Phase 0 + P0 基盤（TASK-000 ~ TASK-003）
- プロジェクトセットアップ
- コアデータモデル
- Protocol定義
- TensorConverter

### Week 3-4: P0 I/O + 特徴量抽出（TASK-004 ~ TASK-006）
- 音声読み込み
- STFT抽出
- ISTFT逆変換

### Week 5-6: P0 データセット（TASK-007 ~ TASK-010）
- HDF5書き込み
- 音声復元
- 基本パイプライン
- CLI

**🎯 Phase 1マイルストーン達成**: End-to-Endで音声→STFT→データセット→復元が動作

### Week 7-8: P1 前処理（TASK-011 ~ TASK-013）
- AudioPreprocessor
- SpectrogramPreprocessor
- Griffin-Lim

### Week 9-10: P1 リアルタイム処理（TASK-014 ~ TASK-015）
- StreamBuffer
- ストリーミング処理

### Week 11: P1 設定・GPU（TASK-016 ~ TASK-017）
- YAML設定管理
- GPU加速

**🎯 Phase 2マイルストーン達成**: リアルタイム処理対応、本番運用可能

### Week 12-13: P2 追加特徴量（TASK-018 ~ TASK-020）
- メルスペクトログラム
- MFCC
- データ拡張

### Week 14: P2 その他（TASK-021 ~ TASK-022）
- データセット分割
- その他音響特徴量

**🎯 Phase 3マイルストーン達成**: 主要特徴量完備

### Week 15-16: P3 拡張機能（TASK-023 ~ TASK-025）
- カスタム抽出器
- 増分更新
- チェックポイント

**🎯 Phase 4マイルストーン達成**: 全機能完成

---

## 品質基準

### コード品質
- [ ] PEP 8準拠（ruff/blackでチェック）
- [ ] Type hints完備（mypy通過）
- [ ] Docstring完備（Google-style）
- [ ] 単体テストカバレッジ80%以上

### 統合テスト
- [ ] End-to-Endパイプライン動作確認
- [ ] 音声復元品質確認（SNR > 30dB）
- [ ] リアルタイム処理レイテンシ確認（< 100ms）
- [ ] GPU/CPU両環境での動作確認

### ドキュメント
- [ ] README: セットアップ手順、基本的な使い方
- [ ] API Documentation: 全公開関数のdocstring
- [ ] チュートリアル: End-to-Endサンプル
- [ ] モジュール構造図

---

## リスク管理

| リスク | 影響 | 対策 |
|--------|------|------|
| GPU環境未整備 | P1遅延 | CPU実装を先行、GPU対応は後回し |
| リアルタイム処理レイテンシ超過 | P1品質低下 | バッファサイズ調整、プロファイリング |
| 音声復元品質不足 | P0品質低下 | ISTFT実装の慎重な検証、テストデータ拡充 |
| メモリ効率問題 | NFR違反 | プロファイリング、メモリ最適化タスク追加 |

---

**ステータス**: タスク分解完了  
**作成日**: 2026-01-20  
**総タスク数**: 26タスク  
**総見積もり工数**: 39.0日（約8週間、1人）  
**次のステップ**: タスク承認後、`/kiro-spec-impl audio-feature-extraction [tasks]` で実装開始
