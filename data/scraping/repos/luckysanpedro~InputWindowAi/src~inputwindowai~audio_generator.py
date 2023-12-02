from dotenv import load_dotenv
from openai import OpenAI
import pygame
from pathlib import Path


assistant_message = " Hello, I am the assistant. I am here to help you."
load_dotenv()

client = OpenAI()


def create_audio_from_text(assistant_message, filename="speech.mp3"):
    speech_file_path = Path(__file__).parent / filename
    audio_response = client.audio.speech.create(
        model="tts-1", voice="echo", input=assistant_message
    )
    audio_response.stream_to_file(speech_file_path)


def play_audio(filename):
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(1000)  # wait one second
    pygame.mixer.music.stop()  # stop the music
    pygame.mixer.music.unload()  # unload the current music


# Create audio from the text in the response
def main(assistant_message):
    create_audio_from_text(assistant_message)
    play_audio("speech.mp3")


if __name__ == "__main__":
    main(assistant_message)
