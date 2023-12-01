from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

response = client.audio.speech.create(
    model="tts-1-hd",
    voice="alloy",
    input="Artificial Intelligence is here... And it is here to stay... But worry not... As a brighter future... is ahead of us.",
)

response.stream_to_file("output.mp3")

