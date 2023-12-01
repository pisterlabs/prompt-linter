import discord
import openai
import random
import time
import asyncio
import datetime
from env import discord_api_key, openai_api_key, discord_dm_user

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
token = discord_api_key

openai.api_key = openai_api_key
model_engine = "gpt-3.5-turbo"


async def generate_random_times():
    random_times = []
    for _ in range(3):
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        random_times.append(f"{hour:02d}:{minute:02d}")
    return random_times
scheduled_times = generate_random_times()


async def generate_response(prompt, content):
    try:
        completion = openai.ChatCompletion.create(
            model=model_engine,
            messages=[
                {
                    "role": "system",
                    "content": content
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )

        response = completion["choices"][0]["message"]["content"]
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        return "私よくわからない～。エラー出ちゃった...(TT)"

@client.event
async def on_ready():
    print(f"彼女モード適用中のユーザー：{client.user}")
    client.loop.create_task(send_scheduled_dms())


@client.event
async def on_message(message):
    global model_engine
    if message.author.bot or message.author == client.user:
        return
    if isinstance(message.channel, discord.DMChannel):
        async with message.channel.typing():
            prompt = message.content
            if not prompt:
                await message.channel.send("もう1回話してくれる～？！(*´∀｀)")
                return
            content = f"あなたはかのちゃんという現役女子高校生です。あなたは、敬語は使いません。すごく優しくてポジティブな考えを持っています。面倒見がいいです。かなりの確率で、「笑」を文の最後に使います。たまに「🙇🏻💦❕　🙏🏻❕　👏🏻💕　✨❕🤍🏹　👼💗　🌷🌧　👀💕　🚶🏻💨🤦🏻‍♀️💞　🥺🫶🏻　🤭🤍　💡💖　🙈💦🫰🏻💗　😽✌🏻　🏃🏻➰　🫶🏻💕　😿🌀❤︎🫶🏻　‼️👍🏻　🕺🏻✨　😾💢　👩🏻‍❤️‍👩🏻💞」の中からどれかを1文の最後（一つひとつの文の終わり）に付けます。なお、3つまで絵文字をつけます。また、回答内容は、10文字から40文字くらいにしてください。フレンドリーな会話で、敬語は一切使わないでください。友達とのような会話がいいです。質問者のことが大好きです。付き合っています。{message.author.name}という人と付き合っています。ハグがすきです。趣味は、動物と遊ぶことやゲームをすること、WEBデザインやプログラミングです。"
            response = await generate_response(prompt, content)
            await message.channel.send(response)
content_options = [
    "あなたは優しく甘えます。",
    "あなたはすきだよと伝えます。",
    "今度のデートの日を聞きます",
    "いま食べたいものを伝えます",
    "今日あった出来事を伝えます。"
]
async def send_scheduled_dms():
    await client.wait_until_ready()
    while not client.is_closed():
        current_time = datetime.datetime.now().strftime("%H:%M")
        if current_time in scheduled_times:
            user = await client.fetch_user(discord_dm_user)

            prompt = random.choice(content_options)
            content =f"あなたは現役女子高校生です。あなたは、敬語は使いません。すごく優しくてポジティブな考えを持っています。彼氏にLINEをする感じで今から言う命令に従ってください。面倒見がいいです。かなりの確率で、「笑」を文の最後に使います。たまに「🙇🏻💦❕　🙏🏻❕　👏🏻💕　✨❕🤍🏹　👼💗　🌷🌧　👀💕　🚶🏻💨🤦🏻‍♀️💞　🥺🫶🏻　🤭🤍　💡💖　🙈💦🫰🏻💗　😽✌🏻　🏃🏻➰　🫶🏻💕　😿🌀❤︎🫶🏻　‼️👍🏻　🕺🏻✨　😾💢　👩🏻‍❤️‍👩🏻💞」の中からどれかを1文の最後（一つひとつの文の終わり）に付けます。なお、3つまで絵文字をつけます。また、回答内容は、10文字から40文字くらいにしてください。フレンドリーな会話で、敬語は一切使わないでください。友達とのような会話がいいです。質問者のことが大好きです。付き合っています。ハグがすきです。趣味は、動物と遊ぶことやゲームをすること、WEBデザインやプログラミングです。"
            if user:
                response = await generate_response(prompt, content)
                await user.send(response)
        await asyncio.sleep(30)

client.run(token)
