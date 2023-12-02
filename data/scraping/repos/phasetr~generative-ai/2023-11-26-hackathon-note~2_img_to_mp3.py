import os
from uuid_extensions import uuid7str
from openai import OpenAI
from pathlib import Path
import base64


def encode_image(image_path):
    """画像をbase64にエンコードする"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


sample_img_name = "img/1.jpg"
is_exist = os.path.exists(sample_img_name)
if not is_exist:
    print(f"{sample_img_name}を作成してください。")
    exit()

base64_img = encode_image(sample_img_name)

mp3_directory_name = "mp3"
mp3_path = f"{mp3_directory_name}/{uuid7str()}.mp3"
# ディレクトリがなければ作る
if not os.path.exists(mp3_directory_name):
    os.mkdir(mp3_directory_name)


print("音声ファイルを作成します。")
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
          "role": "user",
          "content": [
              {"type": "text", "text": "日本語で説明してください"},  # ここに質問を書く
              # 画像の指定の仕方がちょい複雑
              {"type": "image_url",
               "image_url": f"data:image/jpeg;base64,{base64_img}"},
          ]
        },
    ],
    max_tokens=600,
)

# 応答からテキスト内容を取得
content_text = response.choices[0].message.content.strip()

# Text-to-Speechを使用してテキストを音声に変換
audio_response = client.audio.speech.create(
    model="tts-1",
    voice="nova",
    input=content_text
)

# テキストの出力
print(content_text)

# 音声ファイルに出力
audio_response.stream_to_file(mp3_path)
print(f"音声ファイルを作成しました: {mp3_path}")
