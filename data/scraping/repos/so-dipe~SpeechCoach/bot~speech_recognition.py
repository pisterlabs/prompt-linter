from config.config import Config
import tempfile
import openai
from io import BytesIO

config = Config()

openai.api_key = config.OPENAI_APIKEY

def transcribe_ogg(ogg_data: bytes) -> str:
    try:
        # Transcribe the OGG audio directly from memory using Whisper
        transcript = openai.Audio.transcribe("whisper-1", ogg_data, input_format="ogg")["text"]

        return transcript

    except Exception as e:
        # Handle any exceptions that may occur during transcription
        print(f"Error transcribing audio: {e}")
        return ""