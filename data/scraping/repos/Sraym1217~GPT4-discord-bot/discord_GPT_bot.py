import discord
import openai
from openai import OpenAI
import nest_asyncio
import asyncio
import os
from dotenv import load_dotenv
from keep_alive import keep_alive
import logging
import datetime

# 環境変数をロード
load_dotenv()

# Discord BotのトークンとOpenAI APIキーを環境変数から取得
# organization IDを環境変数から取得
TOKEN = os.getenv('DISCORD_TOKEN')
client_open = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))
ORG_ID = os.getenv('OPENAI_ORG_ID')

# Discord Botのクライアントを作成
client = discord.Client(intents=discord.Intents.all())

async def keep_typing(channel):
    while True:
        await channel.typing()
        await asyncio.sleep(4)
        
# Botが起動したときの処理
@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')

# メッセージが送信されたときの処理
@client.event
async def on_message(message):
    timestamp = datetime.datetime.now().isoformat()
    # メッセージがBot自身によるもの、またはスレッド外のメッセージであれば無視
    if message.author == client.user or not isinstance(message.channel, discord.Thread):
        return
    # スレッドの会話履歴を取得
    conversation_history = ""
    async for message_in_thread in message.channel.history(oldest_first=True):
        author = "User" if message_in_thread.author != client.user else "Assistant"
        conversation_history += f"{author}: {message_in_thread.content}\n"

     # 文字数を確認
    char_count = len(conversation_history)
    if char_count > 127900:
        # 文字数が127900を超えている場合、古い会話履歴を削除
        char_to_remove = char_count - 127900  # 保持する文字数を調整
        conversation_history = conversation_history[char_to_remove:]

    typing_task = asyncio.ensure_future(keep_typing(message.channel))

    try:
        # OpenAI APIを使用して、メッセージに対する応答を生成する
        # run_in_executor を使用して同期関数を非同期に実行する
        response = await client.loop.run_in_executor(
            None,  # デフォルトのエグゼキュータを使用
            lambda: client_open.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"{conversation_history} \n Assistant:"},
                ],
            )
        )

        # 応答を送信する
        await message.channel.send(response.choices[0].message.content)
    finally:
        typing_task.cancel()  # タスクをキャンセル

# Discord Botを起動する
keep_alive()
client.run(TOKEN)
