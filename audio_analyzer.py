"""
音声解析・可視化ツール - コア機能
librosaを使用してMP3ファイルを解析し、波形とスペクトログラムを生成
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
import io
from PIL import Image


class AudioAnalyzer:
    """音声解析クラス"""

    def __init__(self):
        """初期化"""
        # Matplotlibの日本語フォント設定
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def load_audio(self, file_path: str, sr: int = 22050):
        """
        音声ファイルを読み込む
        Args:
            file_path: 音声ファイルのパス（MP3, WAV等）
            sr: サンプリングレート（デフォルト: 22050Hz）
        Returns:
            y: 音声信号（numpy配列）
            sr: サンプリングレート
        """
        try:
            # librosaで音声を読み込み
            y, sr = librosa.load(file_path, sr=sr)
            return y, sr
        except Exception as e:
            raise Exception(f"音声ファイル読み込みエラー: {str(e)}")

    def plot_waveform(self, y, sr, figsize=(14, 5)):
        """
        波形を描画
        Args:
            y: 音声信号
            sr: サンプリングレート
            figsize: 図のサイズ
        Returns:
            matplotlib Figure オブジェクト
        """
        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # 時間軸を作成
        time = np.arange(0, len(y)) / sr

        # 波形をプロット
        ax.plot(time, y, linewidth=0.5, alpha=0.8, color='#1f77b4')
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title('Waveform', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, time[-1])

        # Y軸の範囲を調整
        y_max = np.abs(y).max()
        ax.set_ylim(-y_max * 1.1, y_max * 1.1)

        fig.tight_layout()
        return fig

    def plot_spectrogram(self, y, sr, figsize=(14, 6)):
        """
        スペクトログラムを描画
        Args:
            y: 音声信号
            sr: サンプリングレート
            figsize: 図のサイズ
        Returns:
            matplotlib Figure オブジェクト
        """
        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # STFTを計算してスペクトログラムを生成
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # スペクトログラムを表示
        img = librosa.display.specshow(
            S_db,
            sr=sr,
            x_axis='time',
            y_axis='hz',
            ax=ax,
            cmap='viridis'
        )

        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Frequency (Hz)', fontsize=12)
        ax.set_title('Spectrogram', fontsize=14, fontweight='bold')

        # カラーバーを追加
        cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
        cbar.set_label('Amplitude (dB)', fontsize=11)

        fig.tight_layout()
        return fig

    def plot_mel_spectrogram(self, y, sr, figsize=(14, 6)):
        """
        メルスペクトログラムを描画
        Args:
            y: 音声信号
            sr: サンプリングレート
            figsize: 図のサイズ
        Returns:
            matplotlib Figure オブジェクト
        """
        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # メルスペクトログラムを計算
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)

        # メルスペクトログラムを表示
        img = librosa.display.specshow(
            S_db,
            sr=sr,
            x_axis='time',
            y_axis='mel',
            ax=ax,
            cmap='magma'
        )

        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Frequency (Hz)', fontsize=12)
        ax.set_title('Mel Spectrogram', fontsize=14, fontweight='bold')

        # カラーバーを追加
        cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
        cbar.set_label('Amplitude (dB)', fontsize=11)

        fig.tight_layout()
        return fig

    def plot_pitch(self, y, sr, figsize=(14, 6)):
        """
        ピッチ（音高）を描画
        Args:
            y: 音声信号
            sr: サンプリングレート
            figsize: 図のサイズ
        Returns:
            matplotlib Figure オブジェクト
        """
        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # ピッチ（基本周波数）を抽出
        # fmin: 最小周波数 (人間の声の範囲: 約80Hz)
        # fmax: 最大周波数 (人間の声の範囲: 約400Hz、楽器の場合は高めに設定)
        pitches, magnitudes = librosa.piptrack(
            y=y,
            sr=sr,
            fmin=80,
            fmax=1000,
            threshold=0.1
        )

        # 各時間フレームで最も強いピッチを選択
        pitch_values = []
        time_frames = []

        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]

            # ピッチが検出された場合のみ記録（0でない場合）
            if pitch > 0:
                pitch_values.append(pitch)
                time_frames.append(t)

        # 時間軸をフレーム番号から秒に変換
        hop_length = 512  # librosaのデフォルト
        times = librosa.frames_to_time(time_frames, sr=sr, hop_length=hop_length)

        # ピッチをプロット
        if len(pitch_values) > 0:
            ax.plot(times, pitch_values, linewidth=1.5, alpha=0.8, color='#d62728', marker='o', markersize=2)
            ax.set_xlabel('Time (seconds)', fontsize=12)
            ax.set_ylabel('Frequency (Hz)', fontsize=12)
            ax.set_title('Pitch (Fundamental Frequency)', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, times[-1] if len(times) > 0 else 1)

            # Y軸の範囲を見やすく設定
            y_min = max(0, min(pitch_values) - 50)
            y_max = max(pitch_values) + 50
            ax.set_ylim(y_min, y_max)
        else:
            # ピッチが検出されなかった場合
            ax.text(0.5, 0.5, 'No pitch detected',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_xlabel('Time (seconds)', fontsize=12)
            ax.set_ylabel('Frequency (Hz)', fontsize=12)
            ax.set_title('Pitch (Fundamental Frequency)', fontsize=14, fontweight='bold')

        fig.tight_layout()
        return fig

    def get_audio_info(self, y, sr):
        """
        音声ファイルの基本情報を取得
        Args:
            y: 音声信号
            sr: サンプリングレート
        Returns:
            音声情報の辞書
        """
        duration = len(y) / sr
        rms = np.sqrt(np.mean(y**2))
        zero_crossings = np.sum(librosa.zero_crossings(y))

        # ピッチ情報を取得
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=80, fmax=1000, threshold=0.1)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)

        # dBFS: 0 dBFS = full scale (クリッピング直前)。負の値ほど小さい音
        dbfs = 20 * np.log10(rms + 1e-10)

        # Crest Factor: ピーク対RMS比 (dB)。高いほどダイナミクスが大きい（非圧縮）
        crest_factor_db = 20 * np.log10(np.abs(y).max() / (rms + 1e-10))

        # テンポ推定
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # Spectral Centroid: スペクトルの重心周波数。高いほど明るい音色
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

        # Spectral Rolloff: 全エネルギーの85%が含まれる周波数。高いほど高域成分が多い
        spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))

        mm, ss = divmod(int(duration), 60)

        info = {
            "Duration": f"{mm}:{ss:02d} ({duration:.2f}s)",
            "Sample Rate": f"{sr} Hz",
            "Total Samples": f"{len(y):,}",
            "RMS Energy": f"{rms:.6f}",
            "Loudness": f"{dbfs:.2f} dBFS",
            "Crest Factor": f"{crest_factor_db:.2f} dB",
            "Zero Crossings": f"{zero_crossings:,}",
            "Max Amplitude": f"{np.abs(y).max():.6f}",
            "Min Amplitude": f"{np.abs(y).min():.6f}",
            "Tempo": f"{float(tempo):.1f} BPM",
            "Spectral Centroid": f"{spectral_centroid:.2f} Hz",
        }

        # ピッチ統計を追加
        if len(pitch_values) > 0:
            info["Average Pitch"] = f"{np.mean(pitch_values):.2f} Hz"
            info["Pitch Range"] = f"{min(pitch_values):.2f} - {max(pitch_values):.2f} Hz"
        else:
            info["Average Pitch"] = "N/A"
            info["Pitch Range"] = "N/A"

        return info

    def figure_to_image(self, fig):
        """
        matplotlib Figureを画像に変換
        Args:
            fig: matplotlib Figure
        Returns:
            PIL Image
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        # copy() forces immediate pixel loading, detaching the image from the
        # buffer so the buffer can be safely closed (Image.open is lazy).
        img = Image.open(buf).copy()
        buf.close()
        return img

    def analyze_audio(self, file_path: str):
        """
        音声ファイルを総合的に解析
        Args:
            file_path: 音声ファイルのパス
        Returns:
            解析結果の辞書（画像と情報）
        """
        # 音声を読み込み
        y, sr = self.load_audio(file_path)

        # 波形を描画
        waveform_fig = self.plot_waveform(y, sr)
        waveform_img = self.figure_to_image(waveform_fig)
        plt.close(waveform_fig)

        # スペクトログラムを描画
        spectrogram_fig = self.plot_spectrogram(y, sr)
        spectrogram_img = self.figure_to_image(spectrogram_fig)
        plt.close(spectrogram_fig)

        # メルスペクトログラムを描画
        mel_spectrogram_fig = self.plot_mel_spectrogram(y, sr)
        mel_spectrogram_img = self.figure_to_image(mel_spectrogram_fig)
        plt.close(mel_spectrogram_fig)

        # ピッチを描画
        pitch_fig = self.plot_pitch(y, sr)
        pitch_img = self.figure_to_image(pitch_fig)
        plt.close(pitch_fig)

        # 音声情報を取得
        audio_info = self.get_audio_info(y, sr)

        result = {
            "waveform": waveform_img,
            "spectrogram": spectrogram_img,
            "mel_spectrogram": mel_spectrogram_img,
            "pitch": pitch_img,
            "info": audio_info
        }

        return result

    def format_audio_info(self, info: dict) -> str:
        """
        音声情報を整形されたテキストに変換
        Args:
            info: 音声情報の辞書
        Returns:
            整形されたテキスト
        """
        lines = ["=== Audio Information ===\n"]
        for key, value in info.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)


if __name__ == "__main__":
    # テスト用
    analyzer = AudioAnalyzer()
    print("音声解析ツールが初期化されました")
