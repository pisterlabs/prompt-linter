import discord
import openai
from env import discord_api_key, openai_api_key
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
token = discord_api_key

openai.api_key = openai_api_key
model_engine = "gpt-3.5-turbo"


@client.event
async def on_ready():
    print("ヒカマニモード適用中のユーザー：{client.user}")


@client.event
async def on_message(message):
    global model_engine
    if message.author.bot:
        return
    if message.author == client.user:
        return
    if isinstance(message.channel, discord.DMChannel):
        async with message.channel.typing():

            try:
                prompt = message.content
                if not prompt:
                    await msg_typing.delete()
                    await message.channel.send("質問内容ありますか～？絶対にありません。")
                    return

                completion = openai.ChatCompletion.create(
                    model=model_engine,
                    messages=[
                        {
                            "role": "system",
                            "content": f"あなたはヒカマーです。敬語は使いません。回答内容は、10文字から20文字くらいにしてください。「○○なぁ、そうに決まってる」「笑、ゥ。」「やぁりましょう！」「これだ！これだわ！」「何を四天王？」「ナイ！」「○○の域を遥かに超えている」「○○は抜ける」「○○だ、ありがたい」「←ついにおかしくなったw」「許せんなぁ」「○○さんの力をお借りしたいんです！」「ﾁｮｯﾄﾏｯﾃﾁｮｯﾄﾏｯﾃ…」「過去と未来の狭間」などの複数文章から1つ選んで会話します。などの言葉を○○に言葉を当てはめて使用して。"
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                )

                response = completion["choices"][0]["message"]["content"]
                await message.channel.send(response)
            except:
                import traceback
                traceback.print_exc()
                await message.reply("エラーだなぁ、いうまでもない。", mention_author=False)

client.run(token)