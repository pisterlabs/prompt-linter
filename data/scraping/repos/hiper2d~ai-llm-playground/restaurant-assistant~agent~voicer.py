from pathlib import Path

from openai import OpenAI
from playsound import playsound


def play_last_message():
    playsound(Path(__file__).parent / "speech.mp3")


class Voicer:
    def __init__(self):
        self.client = OpenAI()
        self.speech_file_path = Path(__file__).parent / "speech.mp3"

    def text_to_speech(self, input):
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=input
        )
        response.stream_to_file(self.speech_file_path)
        playsound(self.speech_file_path)
