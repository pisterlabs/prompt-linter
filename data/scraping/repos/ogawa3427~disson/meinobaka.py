import os
import re
import io
import sys
import json
import discord
import openai
from openai import OpenAI
aiclient = OpenAI()
import base64
import requests
from logging import getLogger, Formatter, FileHandler, INFO
from PIL import Image, ImageOps
# ロガーの設定
logger = getLogger(__name__)
file_handler = FileHandler("disson.log")
formatter = Formatter("%(asctime)s,%(message)s,", datefmt='%Y-%m-%d_%H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(INFO)

# 設定ファイルの読み込み
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

#リッスン先を設定
args = sys.argv
if args[1] == "t":
    listen_channel = config["test_ch"]
else:
    key = "ch" + args[1]
    listen_channel = config[key]

# OpenAIとDiscordの設定
model = config["model"]
ignore_channel = config["ignore_channel"]
TOKEN = os.getenv('DISCORD_BOT_TOKEN')
openai.api_key = os.getenv('OPENAI_API_KEY')
intents = discord.Intents.all()
client = discord.Client(intents=intents)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

image_path = "saved_image.jpg"
@client.event
async def on_ready():
    logger.info(f'Logged in as {client.user.name} (ID: {client.user.id})')
    print(listen_channel)
    print('------')

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if not str(message.channel.id) in listen_channel:
        return

    print(message)
    
    # メッセージに画像が添付されている場合
   # Discordメッセージに添付されている画像をBase64にエンコード
    if message.attachments:
# Discordから取得した画像のバイトデータを読み込む
        image_bytes = await message.attachments[0].read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # リサイズする最大サイズを設定
        max_size = 1080, 1080
        
        # 画像をリサイズ（アスペクト比を維持）
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # RGBAモードの画像をRGBモードに変換
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # JPEGフォーマットで保存するためのバイトストリームを作成
        # 画像をJPEGフォーマットで保存するためのバイトストリームを作成
        with io.BytesIO() as output:
            image.save(output, format="JPEG")
            data = output.getvalue()
        
        # ローカルファイルシステムに画像を保存
        image_filename = "saved_image.jpg"  # 保存するファイル名
        with open(image_filename, "wb") as f:
            f.write(data)
        
        # 画像をBase64にエンコード
        base64_image = base64.b64encode(data).decode('utf-8')

        print(base64_image)
        
    # ...[your previous code for image handling]...
        print('image')
        base64_image = None
        image_url = message.attachments[0].url  # 最初の添付ファイルのURLを取得
        
        if any(image_url.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']):
            image_bytes = await message.attachments[0].read()  # 画像のバイトデータを取得
            base64_image = base64.b64encode(image_bytes).decode('utf-8')  # base64にエンコード
        
        base64_image = encode_image(image_path)
            # 画像が含まれている場合のテキストは無視して良いならば、この部分を削除
            # text_content = message.clean_content
        content = re.sub(r'<@!?(\d+)>', '', message.content)
    # OpenAIのAPIに送信するペイロードを構築
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": content
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
                ]
                }
            ],
            "max_tokens": 1000
        }

    # APIリクエストのヘッダー
        headers = {
            "Authorization": f"Bearer {openai.api_key}"
        }

    # OpenAIのAPIにリクエストを送信
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_json = response.json()
        print(response_json)

    # 応答からテキストを抽出してDiscordに送信
        answer = response_json['choices'][0]['message']['content'] if 'choices' in response_json else response_json
        await message.channel.send(answer)
        return
    # 画像がない場合はテキストのみで処理
    else:
        if client.user.mentioned_in(message):
            model = config["ogawamodel"]
        else:
            model = config["model"]
        
        print('text')
        # Discordメッセージをクリーンアップ（メンション等を取り除く）
        content = re.sub(r'<@!?(\d+)>', '', message.content)

        # OpenAI APIにメッセージを送信して回答を取得
        response = aiclient.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content}
            ]
        )

        # 応答からテキストを抽出
        answer = response.choices[0].message.content

        # Discordに回答を送信
        await message.channel.send(answer)

# Discordボットを起動
client.run(TOKEN)
