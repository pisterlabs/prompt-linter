#!/usr/bin/env python3

# OpenAIのAPIを叩くだけのDiscord Botを複数並列実行する

# メイン
from discord.ext import commands
import asyncio
import json
import discord
import openai

# JSONファイルから設定を読み込む
with open("config.json") as f:
    config = json.load(f)

# API_KEYを設定する
OPENAI_API_KEY = config["openai_api_key"]


# OpenAIとやり取りするDiscord Botクラス
class OpenAIDiscordBot(commands.Bot):
    def __init__(self, openai_api_key, model, prompt):
        self.openai_api_key = openai_api_key
        self.model = model
        self.prompt = prompt

        # OpenAIのAPIキーを設定
        openai.api_key = self.openai_api_key

        # Discord Botの設定
        intents = discord.Intents.default()
        intents.typing = False  # typingを受け取らないように
        intents.message_content = True

        super().__init__(
            command_prefix="$",  # $コマンド名　でコマンドを実行できるようになる
            case_insensitive=True,  # コマンドの大文字小文字を区別しない
            intents=intents,  # 権限を設定
        )

    # ログインしたらターミナルにログイン通知が表示される
    async def on_ready(self):
        print(f"We have logged in as {self.user}")

    # Botにメンションをした場合、OpenAIに問い合わせる
    async def on_message(self, message):
        if message.author == self.user:
            return

        if self.user in message.mentions:
            # typingを表示
            async with message.channel.typing():
                # 会話履歴を初期化
                conversations = [{"role": "system", "content": self.prompt}]
                # メッセージを会話履歴に追加
                conversations.insert(1, {"role": "user", "content": message.content})

                # メッセージが返信か再帰的に確認し、返信元のメッセージをすべて会話履歴に追加
                while message.reference:
                    message = await message.channel.fetch_message(
                        message.reference.message_id
                    )
                    # 返信がBotの場合はrole:assistant、ユーザーの場合はrole:userとして会話履歴に追加
                    if message.author == self.user:
                        conversations.insert(
                            1, {"role": "assistant", "content": message.content}
                        )
                    else:
                        conversations.insert(
                            1, {"role": "user", "content": message.content}
                        )

                # OpenAIに問い合わせ
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=conversations,
                    max_tokens=2048,
                    temperature=0.8,
                )

            await message.reply(response.choices[0]["message"]["content"])

        await self.process_commands(message)


# Botを起動する非同期関数
async def run_bot(bot_conf):
    MODEL = bot_conf["model"]
    PROMPT = bot_conf["prompt"]
    TOKEN = bot_conf["token"]

    bot = OpenAIDiscordBot(openai_api_key=OPENAI_API_KEY, model=MODEL, prompt=PROMPT)

    # ここでBotを起動する処理を実装する
    await bot.start(TOKEN)


# 各Botの設定を取得し、非同期で実行する
loop = asyncio.get_event_loop()
tasks = [run_bot(bot_conf) for bot_conf in config["bots"]]
loop.run_until_complete(asyncio.gather(*tasks))
