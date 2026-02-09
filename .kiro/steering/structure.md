# プロジェクト構造

## ディレクトリ構成
```
dataset_generator/
├── .kiro/              # AI-DLC仕様管理 (steering, specs, settings)
├── .github/            # GitHub Actions, Issue templates, プロンプト
├── src/dataset_generator/
│   ├── cli/            # CLIエントリーポイント (Click ベース)
│   ├── core/           # データモデル・型定義・コンバーター
│   │   ├── models.py       # AudioData, SpectrogramData (dataclass) ✅
│   │   ├── types.py        # Protocol ベースインターフェース ✅
│   │   └── conversions.py  # TensorConverter (形式変換) ✅
│   ├── features/       # 特徴抽出モジュール
│   │   └── stft.py         # STFTExtractor ✅
│   ├── io/             # 入出力処理
│   │   └── audio_loader.py # AudioFileLoader (soundfile/librosa) ✅
│   ├── transforms/     # 変換・復元処理
│   │   └── inverse.py      # ISTFTReconstructor (Griffin-Lim) ✅
│   ├── pipeline/       # パイプライン統合 (Phase 2)
│   ├── config/         # 設定管理 (Phase 2)
│   └── utils/          # ユーティリティ関数
├── tests/              # テストコード (pytest)
│   ├── core/           # core モジュールのテスト ✅
│   ├── features/       # features モジュールのテスト ✅
│   ├── io/             # io モジュールのテスト ✅
│   ├── transforms/     # transforms モジュールのテスト ✅
│   ├── cli/            # CLI テスト
│   ├── config/         # config テスト
│   ├── pipeline/       # pipeline テスト
│   ├── utils/          # utils テスト
│   ├── conftest.py     # pytest フィクスチャ・設定 ✅
│   └── test_package.py # パッケージ全体テスト ✅
├── .devcontainer/      # Dev Container 設定
├── htmlcov/            # カバレッジレポート (生成物)
├── pyproject.toml      # 依存関係・ツール設定
├── README.md           # プロジェクト説明
└── .python-version     # Python バージョン指定

✅ = 実装完了（TASK-000〜006）
```

## 実装進捗
- **Phase 0**: プロジェクト環境構築 ✅
- **Phase 1 (P0必須)**: コアデータモデル、STFT/ISTFT、AudioLoader ✅ (6/12 タスク完了)
- **Phase 2 (P1高優先度)**: AudioMixer、前処理、HDF5、ストリーミング ⏳ (計画中)
- **Phase 3 (P2中優先度)**: メルスペクトログラム、MFCC、設定管理 ⏳
- **Phase 4 (P3低優先度)**: カスタム機能、拡張 ⏳

## コーディング規約
- **PEP 8準拠** - ruff フォーマッターで自動チェック
- **Type hints必須** - すべての関数・メソッド（戻り値も）
- **Docstring**: Google Style (詳細な引数・戻り値説明）
- **エラーハンドリング**: 独自例外クラス（AudioLoadError など）を定義
- **モジュール分離**: 機能ごとに責務を明確に分離

## 命名規約・インポートパターン
- **Classes**: PascalCase (AudioData, STFTExtractor)
- **Functions/Variables**: snake_case
- **相対インポート**: `from ..core.models import AudioData`
- **パッケージインポート**: `from dataset_generator.core import ...`
- **Protocol**: Loader, Processor など動作を明示

## テスト戦略
- **単体テスト**: 各クラス・関数の動作確認
- **統合テスト**: パイプライン・ワークフロー確認
- **カバレッジ目標**: 80%以上 (pytest-cov で測定)
- **パフォーマンステスト**: pytest-benchmark（STFT/ISTFT の計算量測定）
- **テストデータ**: conftest.py で WAV/FLAC テンポラリファイル生成

## データ検証パターン
- **AudioData.__post_init__**: shape、type、consistency チェック
- **SpectrogramData.__post_init__**: 周波数ビン数・フレーム数の整合性確認
- **負の値ガード**: dB 変換時の安全係数設定

## ドキュメント
- **README**: プロジェクト概要、セットアップ、クイックスタート
- **Docstring**: 各公開APIに記載
- **型情報**: 型ヒント自体がドキュメント（mypy strict モード）
- **テストコード**: 実装例・使用方法の参考

## 拡張ポイント
- **Protocol ベース**: AudioLoader を実装すれば新しい入出力源を追加可能
- **Device サポート**: `device` パラメータで CPU/GPU を切り替え
- **Window 関数**: STFTExtractor で任意の窓関数を指定可能
- **Format 変換**: TensorConverter で新しい出力形式を追加可能
