import os
import openai
from dotenv import load_dotenv

load_dotenv()

# Set up the OpenAI API client
openai.api_key = os.getenv("OPENAI_API_KEY")

file = open("/home/varun/Downloads/TheTimFerrissShow_Eric Cressey.mp3", "rb")
transcription = openai.Audio.transcribe("whisper-1", file)

print(transcription)
