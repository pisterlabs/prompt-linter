import os
from pathlib import Path
from openai import OpenAI


os.environ['OPENAI_API_KEY'] = 'openai api key'

client = OpenAI()

speech_file_path = Path(__file__).parent / "speech.mp3"
response = client.audio.speech.create(
  model="tts-1-hd",
  voice="fable",
  input="I am a gooby goober yeah"
)

response.stream_to_file(speech_file_path)
