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

#init
messages = []
image_flag = False


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

image_path = "saved_image.jpg"


@client.event
async def on_ready():
    logger.info(f'Logged in as {client.user.name} (ID: {client.user.id})')
    print(listen_channel)
    print('------')


    # メッセージに画像が添付されている場合
@client.event
async def on_message(message):
    global messages
    global image_flag
    print(message)
    if message.author == client.user:
        return
    if not str(message.channel.id) in listen_channel:
        return
    #エスケープ
    if message.content.startswith('/'):
        if not message.content.startswith('//'):
            print('command')
            return
        else:
            srawsc = True

    if message.content == "quit":
        print(messages)
        messages = []
        image_flag = False
        await message.channel.send("[System]Conversation cleared.")
        return

    #print(message.content)
    message_adder = {
        "role": "user"
    }
    new_content = []
    # メッセージに画像が添付されている場合
    if message.attachments or image_flag:
        print('image')
        image_flag = True
        base64_images = []

        if not messages:
            message_adder = {}
            message_adder["role"] = "system"
            message_adder["content"] = [
                {
                    "type": "text",
                    "text": "You are a helpful assistant."
                }
            ]
            messages.append(message_adder)

        if message.attachments:
            for attachment in message.attachments:
                # Discordから取得した画像のバイトデータを読み込む
                image_bytes = await attachment.read()
                image = Image.open(io.BytesIO(image_bytes))

                if image.mode == 'RGBA':
                   image = image.convert('RGB')

                max_size = 512, 512
                image.thumbnail(max_size, Image.Resampling.LANCZOS)

                with io.BytesIO() as output:
                    image.save(output, format="JPEG")
                    data = output.getvalue()

            # 画像をBase64にエンコード
                base64_image = base64.b64encode(data).decode('utf-8')
                base64_images.append(base64_image)


                pic_part = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }

                new_content.append(pic_part)

        if message.content:
            text_part = {
                "type": "text",
                "text": re.sub(r'<@!?(\d+)>', '', message.content)
            }
            new_content.append(text_part)

            message_adder["content"] = new_content
            messages.append(message_adder)
            #print(messages)
        
        # OpenAIのAPIに送信するペイロードを構築
            response = aiclient.chat.completions.create(
                model=config["visionmodel"],
                messages=messages,
                max_tokens=1000
            )
        # 応答からテキストを抽出してDiscordに送信
            print(response)
            answer = response.choices[0].message.content
            await message.channel.send(answer)
        # 会話履歴に追加
            message_adder = {}
            message_adder["role"] = "assistant"
            message_adder["content"] = [{"type": "text", "text": answer}]
            messages.append(message_adder)
            print("HELLO")
            return

    # 画像がない場合はテキストのみで処理（以前のコードをそのまま使用）
    else:
        if not messages:
            message_adder = {}
            message_adder["role"] = "system"
            message_adder["content"] = "You are a helpful assistant."
            messages.append(message_adder)

        message_adder = {}
        message_adder["role"] = "user"
        message_adder["content"] = [{"type": "text", "text": re.sub(r'<@!?(\d+)>', '', message.content)}]
        messages.append(message_adder)

               
        if client.user.mentioned_in(message):
            model = config["ogawamodel"]
            print("ogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawaogawa")
        else:
            model = config["model"]

        print('text')
        content = re.sub(r'<@!?(\d+)>', '', message.content)

        # OpenAI APIにメッセージを送信して回答を取得
        response = aiclient.chat.completions.create(
            model=model,
            messages=messages
        )

        # 応答からテキストを抽出
        answer = response.choices[0].message.content

        await message.channel.send(answer)

    #CONTEXTに追加
        message_adder = {}
        message_adder["role"] = "assistant"
        message_adder["content"] = answer
        messages.append(message_adder)
        print(messages)

# Discordボットを起動
client.run(TOKEN)
