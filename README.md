# 音声可視化ツール（Audio Visualizer）

Python + librosaを使用した音声ファイルの波形とスペクトログラム可視化Webツールです。MP3やWAVファイルをアップロードするだけで、美しいビジュアライゼーションが自動生成されます。

## 特徴

- **簡単操作**: ドラッグ&ドロップで音声ファイルをアップロード
- **多様な可視化**: 波形、スペクトログラム、メルスペクトログラム、ピッチの4種類
- **対応フォーマット**: MP3, WAV, FLAC, OGG等、librosa対応の全形式
- **詳細情報**: サンプリングレート、長さ、振幅、ピッチなどの音声データを表示
- **Webベース**: ブラウザで動作する使いやすいインターフェース
- **高品質**: matplotlibによる高解像度の画像生成

## デモ画面イメージ

アップロードした音声ファイルに対して以下が表示されます：

1. **波形（Waveform）**: 時間軸に沿った音声振幅
2. **スペクトログラム（Spectrogram）**: 時間-周波数解析
3. **メルスペクトログラム（Mel Spectrogram）**: 人間の聴覚特性を考慮した表示
4. **ピッチ（Pitch）**: 音高（基本周波数）の時間変化
5. **音声情報**: サンプリングレート、長さ、振幅統計、ピッチ統計など

## インストール

### 必要要件

- Python 3.8以上
- 最低1GBのRAM（推奨2GB以上）

### セットアップ

```bash
# プロジェクトディレクトリに移動
cd audio-visualizer

# 仮想環境の作成（推奨）
python -m venv venv

# 仮想環境の有効化
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

## 使い方

### Webアプリケーションの起動

```bash
python app.py
```

起動後、ブラウザで `http://localhost:7860` にアクセスします。

### 使用手順

1. **音声ファイルアップロード**:
   - ブラウザのファイル選択ボックスをクリック、またはファイルをドラッグ&ドロップ
   - MP3, WAV, FLAC, OGG等に対応

2. **自動解析**:
   - ファイルがアップロードされると自動的に解析が開始されます
   - 数秒で波形とスペクトログラムが表示されます

3. **結果の確認**:
   - 波形グラフで音の振幅パターンを確認
   - スペクトログラムで周波数成分を確認
   - メルスペクトログラムで聴覚的な特性を確認
   - ピッチグラフで音高の変化を確認
   - 音声情報で詳細データを確認

### Pythonコードから直接使用

```python
from audio_analyzer import AudioAnalyzer

# 分析器の初期化
analyzer = AudioAnalyzer()

# 音声ファイルを読み込み
y, sr = analyzer.load_audio("your_audio.mp3")

# 波形を描画
waveform_fig = analyzer.plot_waveform(y, sr)
waveform_fig.savefig("waveform.png")

# スペクトログラムを描画
spectrogram_fig = analyzer.plot_spectrogram(y, sr)
spectrogram_fig.savefig("spectrogram.png")

# メルスペクトログラムを描画
mel_spec_fig = analyzer.plot_mel_spectrogram(y, sr)
mel_spec_fig.savefig("mel_spectrogram.png")

# ピッチを描画
pitch_fig = analyzer.plot_pitch(y, sr)
pitch_fig.savefig("pitch.png")

# 音声情報を取得
info = analyzer.get_audio_info(y, sr)
print(info)
```

### 総合的な解析

```python
from audio_analyzer import AudioAnalyzer

analyzer = AudioAnalyzer()
result = analyzer.analyze_audio("your_audio.mp3")

# 結果の取得
waveform_image = result["waveform"]
spectrogram_image = result["spectrogram"]
mel_spectrogram_image = result["mel_spectrogram"]
pitch_image = result["pitch"]
audio_info = result["info"]

# 画像を保存
waveform_image.save("waveform.png")
spectrogram_image.save("spectrogram.png")
mel_spectrogram_image.save("mel_spectrogram.png")
pitch_image.save("pitch.png")

# 情報を表示
print(analyzer.format_audio_info(audio_info))
```

## 技術詳細

### 使用ライブラリ

- **librosa**: 音声信号処理（読み込み、STFT、メルスペクトログラム等）
- **matplotlib**: グラフと可視化
- **Gradio**: Webインターフェース
- **NumPy**: 数値計算
- **Pillow**: 画像処理

### 可視化の詳細

#### 波形（Waveform）
- 時間軸に沿った音声信号の振幅を表示
- 音の大きさや周期性を視覚的に確認
- サンプリング: デフォルト22050Hz

#### スペクトログラム（Spectrogram）
- Short-Time Fourier Transform (STFT) を使用
- 横軸: 時間
- 縦軸: 周波数（Hz）
- 色: 振幅（dB）
- 音の周波数成分の時間変化を可視化

#### メルスペクトログラム（Mel Spectrogram）
- 人間の聴覚特性（メル尺度）を考慮
- 低周波数帯域をより詳細に表示
- 128メルバンド（デフォルト）
- 音声認識、音楽情報検索などに最適

#### ピッチ（Pitch / Fundamental Frequency）
- `librosa.piptrack()`を使用して基本周波数を抽出
- 横軸: 時間（秒）
- 縦軸: 周波数（Hz）
- 音の高さの時間変化を可視化
- 歌声の音程変化や楽器の演奏分析に最適
- デフォルト検出範囲: 80Hz〜1000Hz

### 音声情報

以下の情報が自動計算されます：

- **Duration**: 音声の長さ（秒）
- **Sample Rate**: サンプリングレート（Hz）
- **Total Samples**: サンプル総数
- **RMS Energy**: 二乗平均平方根エネルギー
- **Zero Crossings**: ゼロ交差数（音の周波数の指標）
- **Max/Min Amplitude**: 最大/最小振幅
- **Average Pitch**: 平均ピッチ（Hz）
- **Pitch Range**: ピッチの範囲（最小〜最大Hz）

## カスタマイズ

### サンプリングレートの変更

デフォルトは22050Hzですが、変更可能です：

```python
# より高品質（高周波数まで解析）
y, sr = analyzer.load_audio("audio.mp3", sr=44100)

# より軽量（処理速度優先）
y, sr = analyzer.load_audio("audio.mp3", sr=16000)

# 元のサンプリングレートを維持
y, sr = analyzer.load_audio("audio.mp3", sr=None)
```

### 図のサイズ変更

```python
# 波形の図のサイズを変更（幅, 高さ）インチ単位
waveform_fig = analyzer.plot_waveform(y, sr, figsize=(20, 8))

# スペクトログラムのサイズ変更
spectrogram_fig = analyzer.plot_spectrogram(y, sr, figsize=(16, 10))
```

### メルバンド数の変更

`audio_analyzer.py`の`plot_mel_spectrogram()`メソッドで調整：

```python
# より詳細な解析
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256)

# より軽量な解析
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
```

### ピッチ検出範囲の変更

`audio_analyzer.py`の`plot_pitch()`メソッドで周波数範囲を調整：

```python
# 女性の声に最適化（高めの周波数）
pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=150, fmax=500)

# 楽器の広い範囲をカバー
pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=80, fmax=2000)

# 男性の低い声に最適化
pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=60, fmax=300)
```

## 対応フォーマット

librosaがサポートする全ての音声フォーマットに対応：

- **MP3** (.mp3)
- **WAV** (.wav)
- **FLAC** (.flac)
- **OGG** (.ogg)
- **M4A** (.m4a)
- その他多数

## トラブルシューティング

### MP3ファイルが読み込めない

MP3のデコードには `ffmpeg` または `audioread` が必要です：

```bash
# ffmpegをインストール
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS (Homebrew):
brew install ffmpeg

# Windows:
# https://ffmpeg.org/download.html からダウンロード

# または、audioreadをインストール
pip install audioread
```

### メモリエラー

長い音声ファイルの場合、メモリ不足が発生する可能性があります：

```python
# 音声の一部だけを読み込む（最初の30秒）
y, sr = librosa.load("audio.mp3", duration=30)

# オフセットを指定（10秒目から20秒間）
y, sr = librosa.load("audio.mp3", offset=10, duration=20)
```

### Gradioがブラウザで開かない

手動でブラウザを開き、以下のURLにアクセス：
```
http://localhost:7860
```

または、ポートが使用中の場合は `app.py` の `server_port` を変更：

```python
demo.launch(
    server_port=7861,  # 別のポート番号に変更
    # ...
)
```

## 応用例

### 音楽分析
- テンポやリズムパターンの可視化
- 楽器の周波数特性の分析
- ミックスのバランス確認
- **ピッチ分析による音程の確認**
- **メロディラインの可視化**

### 音声認識
- 発話パターンの分析
- 音素の可視化
- ノイズの検出
- **声の高さ変化の分析**
- **イントネーションの研究**

### 音響研究
- 周波数特性の解析
- 音質評価
- 音響イベントの検出
- **基本周波数の時系列分析**

## パフォーマンス

### 処理時間の目安

- 3分のMP3ファイル（44.1kHz）: 約2-5秒
- 5分のMP3ファイル（44.1kHz）: 約3-7秒

※コンピュータのスペックによって変動します

### 最適化のヒント

1. サンプリングレートを下げる（22050Hz以下）
2. 短い音声クリップで分析
3. メルバンド数を減らす（64以下）

## ライセンス

本ツールはMITライセンスの下で公開されています。

使用ライブラリのライセンス：
- librosa: ISC License
- matplotlib: PSF License
- Gradio: Apache 2.0

## 貢献

バグ報告や機能提案は、GitHubのIssuesにてお願いします。

## 参考資料

- [librosa公式ドキュメント](https://librosa.org/doc/latest/index.html)
- [スペクトログラムについて](https://en.wikipedia.org/wiki/Spectrogram)
- [メル尺度について](https://en.wikipedia.org/wiki/Mel_scale)

---

**開発**: Audio Visualizer Project
**バージョン**: 1.0.0
**更新日**: 2025年
