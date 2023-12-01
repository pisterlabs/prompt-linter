import os
from os import path
import speech_recognition as sr
from openai import OpenAI
import pygame

# Initialize the recognizer
recognizer = sr.Recognizer()

# Set the microphone as the audio source
microphone = sr.Microphone()

pygame.mixer.init()

voice_client = OpenAI()
api_key = os.environ['OPENAI_API_KEY']
speech_file_path = path.join(path.curdir, "speech.mp3")

# Adjust for ambient noise
with microphone as source:
    print("Adjusting for ambient noise. Please wait...")
    recognizer.adjust_for_ambient_noise(source)
    print("Ambient noise adjustment complete.")

def human_voice_output(text):
    print("Saying ", text)
    # wait until we have said all we had to say
    while pygame.mixer.music.get_busy():
      pygame.time.Clock().tick(10)

    response = voice_client.audio.speech.create(
        model="tts-1",
        voice="shimmer",
        input=text
    )
    response.stream_to_file(speech_file_path)
    pygame.mixer_music.load(speech_file_path)
    pygame.mixer_music.play() 

  
def human_voice_input(question) -> str:
    human_voice_output(question)

    while pygame.mixer.music.get_busy():
      pygame.time.Clock().tick(10)
    with microphone as source:
      audio_data = recognizer.listen(source, timeout=10)
      transcription = recognizer.recognize_whisper_api(audio_data, api_key=api_key)
      print("Hearing ", transcription.text)
      return transcription.text
