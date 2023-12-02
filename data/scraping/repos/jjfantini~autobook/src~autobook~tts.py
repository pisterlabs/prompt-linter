import os
from typing import Literal

from openai import OpenAI

from autobook.core.env import Env


class SpeechGenerator:
    def __init__(self, api_key: str, model: str = "tts-1", voice="nova"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.voice = voice

    def generate_speech(self, text, file_path):
        response = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text
        )
        response.stream_to_file(file_path)


def main():
    api_key = Env().OPENAI_API
    speech_generator = SpeechGenerator(api_key)

    text = "Okay! What time do you want to start making dinner? 5 mins?"
    speech_file_path = "test.mp3"

    speech_generator.generate_speech(text, speech_file_path)
    print("----Success----")


if __name__ == '__main__':
    main()