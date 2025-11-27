"""
音声可視化Webツール - Gradioアプリケーション
MP3などの音声ファイルをアップロードして波形とスペクトログラムを表示
"""

import gradio as gr
from audio_analyzer import AudioAnalyzer
import tempfile
import os


def analyze_audio_file(audio_file):
    """
    音声ファイルを解析してビジュアライゼーションを生成
    Args:
        audio_file: アップロードされた音声ファイル
    Returns:
        波形画像、スペクトログラム画像、メルスペクトログラム画像、音声情報テキスト
    """
    if audio_file is None:
        return None, None, None, "音声ファイルをアップロードしてください。"

    try:
        # 分析器を初期化
        analyzer = AudioAnalyzer()

        # 音声ファイルのパスを取得
        audio_path = audio_file

        # 音声を解析
        result = analyzer.analyze_audio(audio_path)

        # 音声情報を整形
        info_text = analyzer.format_audio_info(result["info"])

        return (
            result["waveform"],
            result["spectrogram"],
            result["mel_spectrogram"],
            info_text
        )

    except Exception as e:
        error_msg = f"エラーが発生しました:\n{str(e)}\n\n詳細: {type(e).__name__}"
        return None, None, None, error_msg


def create_demo():
    """Gradioデモを作成"""

    with gr.Blocks(title="音声可視化ツール") as demo:
        gr.Markdown(
            """
            # 音声可視化ツール（Audio Visualizer）

            ## 機能
            - **MP3/WAV対応**: 様々な音声フォーマットに対応
            - **波形表示**: 時間軸に沿った音声振幅の可視化
            - **スペクトログラム**: 時間-周波数解析による音声の周波数成分表示
            - **メルスペクトログラム**: 人間の聴覚特性を考慮した周波数表示
            - **音声情報**: サンプリングレート、長さ、振幅などの詳細情報

            ## 使い方
            1. 下のボックスに音声ファイル（MP3, WAV等）をアップロード
            2. 自動的に解析が開始され、波形とスペクトログラムが表示されます
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="音声ファイル",
                    type="filepath",
                    sources=["upload"]
                )

                gr.Markdown(
                    """
                    ### 対応フォーマット
                    - MP3
                    - WAV
                    - FLAC
                    - OGG
                    - その他librosa対応形式
                    """
                )

            with gr.Column(scale=1):
                info_output = gr.Textbox(
                    label="音声情報",
                    lines=10
                )

        gr.Markdown("---")
        gr.Markdown("## ビジュアライゼーション")

        with gr.Row():
            waveform_output = gr.Image(
                label="波形（Waveform）",
                type="pil"
            )

        with gr.Row():
            spectrogram_output = gr.Image(
                label="スペクトログラム（Spectrogram）",
                type="pil"
            )

        with gr.Row():
            mel_spectrogram_output = gr.Image(
                label="メルスペクトログラム（Mel Spectrogram）",
                type="pil"
            )

        # 音声がアップロードされたら自動的に解析
        audio_input.change(
            fn=analyze_audio_file,
            inputs=[audio_input],
            outputs=[
                waveform_output,
                spectrogram_output,
                mel_spectrogram_output,
                info_output
            ]
        )

        gr.Markdown(
            """
            ---
            ### 技術情報
            - **解析ライブラリ**: librosa
            - **可視化**: matplotlib
            - **サンプリングレート**: 22050 Hz（デフォルト）

            ### 各ビジュアライゼーションについて

            #### 波形（Waveform）
            時間軸に沿った音声信号の振幅を表示します。音の大きさや周期的なパターンを確認できます。

            #### スペクトログラム（Spectrogram）
            時間-周波数領域での音声の特性を表示します。横軸が時間、縦軸が周波数、色が振幅を表します。
            音楽の音符や声の高さの変化を視覚的に確認できます。

            #### メルスペクトログラム（Mel Spectrogram）
            人間の聴覚特性を考慮したスペクトログラムです。低周波数帯域をより詳細に表示します。
            音声認識や音楽情報検索などで広く使用されています。
            """
        )

    return demo


if __name__ == "__main__":
    print("音声可視化ツールを起動中...")
    print("ブラウザが自動的に開きます...")

    demo = create_demo()

    # アプリケーション起動
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
