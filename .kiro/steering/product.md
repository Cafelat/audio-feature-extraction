# プロダクト要件

## プロダクト概要
音声データから特徴量を抽出し、機械学習用のデータセットを生成するツール

## 目的
- 音声データの前処理と特徴量抽出を自動化
- 機械学習モデルのトレーニングに適したデータセット形式で出力
- 研究・開発における音声処理ワークフローの効率化

## ターゲットユーザー
- 音声認識・音声合成の研究者
- 機械学習エンジニア
- データサイエンティスト

## コアバリュー
- **柔軟な特徴量抽出パイプライン** - 複数の出力形式（COMPLEX、MAGNITUDE_PHASE、MAGNITUDE_PHASE_TRIG、MAGNITUDE_ONLY）をサポート
- **GPU加速による高速処理** - PyTorchベースのGPU対応実装
- **完全な可逆変換** - STFT/ISTFTと位相情報保持による完全復元
- **robust な音声復元** - 振幅のみからのGriffin-Lim逆変換対応
- **型安全性** - Protocol ベースのインターフェース定義と完全な型ヒント

## 実装済み主要機能
- **AudioData/SpectrogramData**: 音声・スペクトログラムの検証付きデータモデル
- **STFTExtractor**: GPU対応の STFT 特徴量抽出
- **ISTFTReconstructor**: スペクトログラムから音声への逆変換
- **TensorConverter**: numpy/torch テンソル間の形式変換
- **AudioFileLoader**: WAV/FLAC 読み込み、リサンプリング対応
- **Protocol ベースの型チェック**: AudioLoader プロトコルによる拡張性

## データモデル設計原則
- 各モデルは初期化時にデータ検証（shape、type、consistency チェック）を実施
- メタデータ辞書で拡張可能な構造
- numpy/torch の両テンソル形式をサポート
