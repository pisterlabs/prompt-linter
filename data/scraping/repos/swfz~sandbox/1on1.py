import numpy as np
import json
import os
import soundcard as sc
import threading
import queue
import soundfile as sf
import openai
from pprint import pprint

# moved. swfz/gpt-1on1

# 録音の設定
samplerate = 48000  # サンプルレート
threshold = 0.1  # 無音のしきい値
silence_duration = 2  # 無音の継続時間 (秒)

# 無音判定のためのサンプル数
silence_samples = int(samplerate * silence_duration)

# 録音データ取得
microphone = sc.default_microphone()

print('[Info] 1on1 Start!')

def transcribe(filename):
    audio_file = open(filename, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file, language='ja')

    return transcript['text']

messages=[]

def chatgpt(text):
    first_prompt = """
あなたは優秀なエンジニアリングマネージャーであり私のメンターです
「テキスト」以下の設問に対し以下の条件で返答してください
- 深掘りして私に質問し相談の状況の詳細を話させてより具体的な話をしてください
- 回答は3つまでとしてください
- 確実な解決策がある場合は解決策を提示してください
- 5回以上やり取りした結果、総合すると私はこういう人だねという特徴をフィードバックしてください
テキスト:

"""

    if len(messages) == 0:
        payload_text = first_prompt + text
    else:
        payload_text = text

    prompt = {"role": "user", "content": payload_text}

    messages.append(prompt)
    print('[Info] GPT Thinking...............')
    res = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

    role = res.choices[0]["message"]["role"]
    content = res.choices[0]["message"]["content"]

    messages.append({"role": role, "content": content})
    print(f"{role}: {content}")


def record_audio(queue):
    with microphone.recorder(samplerate=samplerate) as mic:
        while True:
            data = mic.record(numframes=silence_samples)
            queue.put(data)
            if stop_recording.is_set():
                break

# ストリームを処理するスレッド
stop_recording = threading.Event()
q = queue.Queue()
record_thread = threading.Thread(target=record_audio, args=(q,))
record_thread.start()

# 録音データをリアルタイムで処理
segments = []
current_data = np.array([], dtype=np.float32)
silent_frames = 0
outfilename="segments.wav"

try:
    while True:
        new_data = q.get()
        flatten = np.frombuffer(new_data, dtype=np.float32)

        current_data = np.concatenate((current_data, flatten))

        for sample in current_data:
            if np.abs(sample) < threshold:
                silent_frames += 1
            else:
                silent_frames = 0

            if silent_frames >= silence_samples:
                splitted = current_data[:-silent_frames]

                # リセット
                current_data = np.array([], dtype=np.float32)
                silent_frames = 0

                # 無音が続いた場合は切り上げ
                if not np.all(np.abs(splitted)<threshold):
                    segments.append(splitted)
                    print(f"Segment {len(segments)} detected")  # ここで区切られたセグメントを表示

                    file_data = np.reshape(splitted, [-1,1])
                    filename = f"./wavs/{len(segments)}-{outfilename}"
                    sf.write(file=filename, data=file_data, samplerate=samplerate)

                    transcribed_text = transcribe(filename)
                    print(transcribed_text)

                    chatgpt(transcribed_text)


except KeyboardInterrupt:
    stop_recording.set()
    record_thread.join()

    # 最後の部分を追加
    if len(current_data) > 0:
        segments.append(current_data)

# 結果を表示
for i, segment in enumerate(segments):
    print(f"Segment {i + 1}: {segment}")

for i, m in enumerate(messages):
    print(f"[{i}] {m['role']}: {m['content']}")

with open(f"logs/1on1-{os.getpid()}.json", "w") as f:
    json.dump(messages, f)
