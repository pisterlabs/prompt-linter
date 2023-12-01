import sounddevice as sd
from scipy.io.wavfile import write
import openai
import numpy as np
import threading
import time

# OpenAIのAPIキーを設定
openai.api_key = 'your-api-key'

# 録音のパラメータ
fs = 44100  # サンプルレート
recording = np.array([])  # 録音データを保存する配列

# 録音の開始と終了を制御するフラグ
is_recording = False

def record():
    """録音を行う関数"""
    global is_recording
    global recording
    while True:
        if is_recording:
            # 録音中の場合、0.5秒分の録音データを追加
            recording_chunk = sd.rec(int(0.5 * fs), samplerate=fs, channels=1)
            sd.wait()
            recording = np.append(recording, recording_chunk)
        else:
            # CPU負荷を下げるために1ミリ秒待機
            time.sleep(0.001)

# 録音スレッドの開始
recording_thread = threading.Thread(target=record)
recording_thread.start()

while True:
    if not is_recording:
        input("Enterキーを押すと録音が開始します。\n")
        # 録音を開始
        is_recording = True
        print("録音を開始します。\n")
    else:
        input("録音中です。Enterを押すと録音が終了します。\n")
        # 録音を終了
        is_recording = False
        print("録音が終了しました。")
        if recording.size > 0:
            # 録音データが存在する場合、データをファイルに保存
            write('output.wav', fs, recording)

            # ファイルをバイナリモードで開く
            with open('output.wav', "rb") as audio_file:
                # Whisper APIを使用してオーディオファイルをテキストに変換
                transcript = openai.Audio.transcribe("whisper-1", audio_file)

            # 音声からテキスト変換した結果を表示
            print("\n音声認識結果: {}\n".format(transcript.text))

            # 録音データをリセット
            recording = np.array([])
