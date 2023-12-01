from decouple import config
from pathlib import Path
from openai import OpenAI


client = OpenAI(api_key=config("API_KEY"))

for count in range(10):

    speech_file_path = Path(__file__).parent / "out_speech.mp3"
    response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input="Send reinforcements we are going to advance."
    )

    response.stream_to_file(speech_file_path)

    audio_file = open("out_speech.mp3", "rb")
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )

    print(transcript)
