from dotenv import load_dotenv
import time
import os
from sys import platform

import openai
import speech_recognition as sr
import whisper
import pyaudio
import wave

class GScribe:
    def __init__(self) -> None:
        pass

    def transcribe(self, filename):
        """Transcribes audio to text using Google's Speech Recognition API"""
        recognizer = sr.Recognizer()
        with sr.AudioFile(filename) as source:
            audio_data = recognizer.record(source)
        
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except Exception as e:
            # print("[GScribe] An error occurred: {}".format(e))
            print("...") 


def main():
    # Example usage
    gscribe = GScribe()
    text = gscribe.transcribe("output.wav")
    print(text)

if __name__ == "__main__":
    main()