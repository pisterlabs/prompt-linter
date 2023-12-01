from openai import AsyncOpenAI, OpenAI
import os
import io
from pydub import AudioSegment
from pydub.playback import play
import asyncio

from gpt import GPT

client = OpenAI(
    api_key=os.getenv("OPENAI_KEY"),
)

ai = GPT()


# total = ''
# for chunk in ai.chats(input()):
#     total += chunk
#     if len(total) > 10:
# async def respond():
#     stream = await client.chat.completions.create(
#         model="gpt-4-1106-preview",
#         messages=[{"role": "user", "content": "Tell me a short story"}],
#         stream=True,
#     )
#     async for part in stream:
#         print(part.choices[0].delta.content or "", end='')


response = client.audio.speech.create(
    model="tts-1-hd",
    voice="nova",
    input=ai.chat("tell me a two sentence bedtime story"),
    speed=1
)

byte_stream = io.BytesIO(response.content)
audio = AudioSegment.from_file(byte_stream, format="mp3")
play(audio)

# asyncio.run(respond())