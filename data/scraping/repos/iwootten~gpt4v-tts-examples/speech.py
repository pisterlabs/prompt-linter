from openai import OpenAI
from pathlib import Path

import typer

client = OpenAI()

def say(text: str):
    speech_file_path = Path(__file__).parent / "data" / "speech.mp3"

    response = client.audio.speech.create(
        model="tts-1",
        voice="fable",
        input=text
    )

    response.stream_to_file(speech_file_path)


if __name__ == "__main__":
    typer.run(say)