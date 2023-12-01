import json
import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def voice_to_text(file):
    audio_file = open(file, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    json_obj = json.loads(str(transcript))
    text = json_obj['text']
    print(text)
    return text
