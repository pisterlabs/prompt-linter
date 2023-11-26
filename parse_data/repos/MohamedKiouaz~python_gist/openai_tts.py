from pathlib import Path
from openai import OpenAI
client = OpenAI()

speech_file_path = Path(__file__).parent / "speech.mp3"
response = client.audio.speech.create(
    model="tts-1",
    voice="nova",
    input="Déjà c'est cool si globalement ça c'est bien passé 👌😁"
)

response.stream_to_file(speech_file_path)