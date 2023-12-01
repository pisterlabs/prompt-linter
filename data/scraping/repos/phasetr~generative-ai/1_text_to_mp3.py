import os
from uuid_extensions import uuid7str
from openai import OpenAI
from pathlib import Path

sample_text_name = "sample1.txt"
is_exist = os.path.exists(sample_text_name)
if not is_exist:
    print(f"{sample_text_name}を作成してください。")
    exit()

print("テキストを読み込みます。")
text = ""
with open(sample_text_name, mode="r", encoding="utf-8") as f:
    text = f.read()

print(text)

mp3_directory_name = "mp3"
mp3_path = f"{mp3_directory_name}/{uuid7str()}.mp3"
# ディレクトリがなければ作る
if not os.path.exists(mp3_directory_name):
    os.mkdir(mp3_directory_name)

print("音声ファイルを作成します。")
client = OpenAI()
response = client.audio.speech.create(
    model="tts-1",
    voice="nova",
    input=text
)
response.stream_to_file(mp3_path)
print(f"音声ファイルを作成しました: {mp3_path}")
