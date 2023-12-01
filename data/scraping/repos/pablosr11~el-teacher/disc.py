"""Discord bot to transcribe voice notes and reply with feedback"""
import asyncio
import os
import random
import string

import discord
import httpx
import openai
import tiktoken_async
from dotenv import load_dotenv
from upstash_redis.asyncio import Redis

load_dotenv()
DEFAULT_CONVERSATION_CREDITS = 100
openai.api_key = os.getenv("OPENAI_API_KEY")
redis = Redis(
    url=os.getenv("REDIS_URI"),
    token=os.getenv("REDIS_PASS"), 
    allow_telemetry=False
    )

class MyClient(discord.Client):
    """Discord client"""

    async def on_member_join(self, member:discord.Member):
        """Handle member join event"""
        channel = await member.create_dm()
        await channel.send("Welcome to the server! Send me a voice note about a topic and I will try to provide feedback")
        await asyncio.sleep(0.1)
        await self.power.send(f"***new_joiner***: {member.name}")
        return

    async def on_ready(self):
        """Handle client ready event"""
        print(f"Logged in as {self.user} (ID: {self.user.id})")
        print("------")
        self.power = await client.fetch_user(os.getenv("POWERUSER_ID"))

    async def on_message(self, message: discord.Message):
        """Handle incoming messages"""


        # how to enable people to talk to bot directly without joining a server
        # generate headline and store in redis along with credits.

        # stop bot from replying to itself
        if message.author == client.user:
            return

        IS_AUDIO = len(message.attachments) > 0
        author = message.author.name
        channel = message.channel

        await self.power.send(f"***new***: {author} | ***is_audio***: {IS_AUDIO} | ***channel***: {channel}")
        await asyncio.sleep(0.1)

        if "Direct Message" not in str(channel):
            await message.channel.send("I only reply in DMs")
            return

        if len(message.attachments) > 0:
            url = message.attachments[0].url

            author = message.author.name       
            conv_left = await redis.decr(author)

            if conv_left == -1: # doesnt exist
                await redis.set(author, DEFAULT_CONVERSATION_CREDITS)
                conv_left = DEFAULT_CONVERSATION_CREDITS


            if not conv_left or conv_left == 0:
                await message.channel.send("You have no more conversations left. We will be in touch. Alternatively, send a small message to https://psiesta.com")
                await self.power.send(f"Out of credits: {author}\n*** SET {author} {DEFAULT_CONVERSATION_CREDITS}***\n")
                return

            await message.channel.send("Thanks for your message. Your feedback is being generated...")
    
            async with httpx.AsyncClient() as httpx_client:
                resp = await httpx_client.get(url)
                b_content = resp.content

            filename = (
                "".join(
                    random.choice(string.ascii_uppercase + string.digits)
                    for _ in range(6)
                )
                + ".ogg"
            )

            ## turn this into async
            with open(filename, "wb") as ff:
                ff.write(b_content)
                ff.seek(0)

            audio_file = open(filename, "rb")
            trans = await openai.Audio.atranscribe("whisper-1", audio_file)
            audio_file.close()
            os.remove(filename)
            transcript = trans["text"]

            if transcript:
                # count tokens async
                encoder = await tiktoken_async.encoding_for_model("gpt-3.5-turbo")
                n_tokens = await client.loop.run_in_executor(
                    None, encoder.encode, transcript
                )
                if len(n_tokens) < 10:
                    await message.channel.send(
                        "Audio message has to be longer. We could not process your message."
                    )
                    return

                msg = transcript

                completion = await openai.ChatCompletion.acreate(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful teacher, judge, assistant. With a lot of life experiences. \
                                People come to you with speeches and explanations and questions \
                                and you provide helpful feedbackt. From \
                                mistakes in the facts, to the structure of the speech to any other suggestions. \
                                If possible you suggest other related topics to learn \
                                too. You always reply in the language of the speech.",
                        },
                        {"role": "user", "content": msg},
                    ],
                )
                respuesta = completion.choices[0].message["content"]
                await message.channel.send(respuesta.encode("utf-8").decode("utf-8"))
            else:
                await message.channel.send("I can't understand that, please try again.")
                return
        else:
            await message.channel.send("I only reply to voice notes")


intents = discord.Intents.default()
intents.message_content = True  # pylint: disable=locally-disabled, multiple-statements, assigning-non-slot, line-too-long
intents.members = True # pylint: disable=locally-disabled, multiple-statements, assigning-non-slot, line-too-long
client = MyClient(intents=intents)
client.run(os.getenv("DISCORD_TOKEN"))
