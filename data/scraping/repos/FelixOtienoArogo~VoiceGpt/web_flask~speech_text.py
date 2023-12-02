"""Converting the speech to text."""
from openai import OpenAI
import os
client = OpenAI()

#Getting the file path
#SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) #path of this file
#UPLOADS_DIR = os.path.join(SCRIPT_DIR, 'uploads')
#mp3_file_path = os.path.join(UPLOADS_DIR, 'audio.mp3')

def speech_text(mp3_file_path):
    audio_file=open(mp3_file_path, "rb")
    transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text")
    return transcript
