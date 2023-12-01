import discord
import os
from openai import OpenAI

# Intentsの設定
intents = discord.Intents.default()
intents.messages = True
intents.reactions = True

# Discord ClientをIntentsと共に初期化
client = discord.Client(intents=intents)

# 環境変数からDiscord BOTのトークンを取得
discord_bot_token = os.environ['DISCORD_BOT_TOKEN']

# 反応する絵文字を設定
target_emoji = "🅾"

@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')

@client.event
async def on_reaction_add(reaction, user):
    # 指定した絵文字に反応する場合の処理
    if str(reaction.emoji) == target_emoji:
        message = reaction.message
        print(f'detected! :{message}')
        openai_client = OpenAI(os.environ['GPT_API_KEY'])  # OpenAIクライアントの初期化
        completion = openai_client.chat.completions.create(
            model="gpt-4-0314",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message.content}
            ]
        )
        await message.channel.send(completion.choices[0].message)

# ボットを実行
client.run(discord_bot_token)
