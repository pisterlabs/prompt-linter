import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

path = Path("../Environment-Variables/.env")
load_dotenv(dotenv_path=path)

# Set up openai client
openai = OpenAI(
    organization=os.getenv('organization'),
    api_key=os.getenv("api_key")
)


# transcription function
def transcribe(file_path):
    # Open audio file
    audio_file = open(file_path + '\output_audio.mp3', "rb")

    # Use Whisper-1 model to transcribe audio file
    transcription = openai.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )

    # Write output to txt file in root directory
    with open('transcription.txt', 'w') as f:
        f.write(transcription['text'])
