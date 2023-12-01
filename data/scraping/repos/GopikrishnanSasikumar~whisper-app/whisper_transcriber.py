from openai import OpenAI
import dotenv
import os

dotenv.load_dotenv("ops/.env")

def transcribe_audio(audio_path, model="whisper-1"):
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    audio_file = open(audio_path, "rb")
    transcript = client.audio.transcriptions.create(
        model=model,
        file=audio_file
    )
    text = transcript.text
    return text



