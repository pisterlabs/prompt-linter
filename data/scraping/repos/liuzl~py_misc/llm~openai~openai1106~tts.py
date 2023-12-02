from pathlib import Path
import openai
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")

text = """The quick brown fox jumped over the lazy dog."""
text = '''这张图片显示了一条穿过高草地的木质栈道。栈道在草丛中伸展，直至远方，两旁的草地看上去非常繁茂。背景中是一些树木和一个多云的天空，给人一种宁静和自然的感觉。天空是晴朗的，有几朵白云点缀其中。整体上，这张图片给人一种平和、宁静的田园风光。'''

speech_file_path = Path(__file__).parent / "speech1.mp3"
response = openai.audio.speech.create(
  model="tts-1",
  voice="nova", #"alloy",
  input=text,
)
response.stream_to_file(speech_file_path)
