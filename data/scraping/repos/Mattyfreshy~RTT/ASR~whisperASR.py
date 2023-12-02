from dotenv import load_dotenv
import time
import os
from sys import platform

import openai
import speech_recognition as sr
import whisper
import pyaudio
import wave
import torch

# Load .env file
load_dotenv()

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")


""" Whisper Model """
class Wisp:
    def __init__(self, model='base', microphone=sr.Microphone()):
        """ 
        If no model is specified, the default is 'base'
        Whisper model sizes ['tiny', 'base', 'small', 'medium', 'large']
        """
        self.microphone = microphone
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fp16 = True if self.device == 'cuda' else False
        self.model = whisper.load_model(model, device=self.device)

    def load_model(self, model: str):
        """
        Loads a model from Whisper.
        Whisper model sizes ['tiny', 'base', 'small', 'medium', 'large']
        Default is 'base'
        """

        print("\033[32mLoading Whisper Model...\033[37m")
        self.model = whisper.load_model(model if model else 'base', device=self.device)

    def generate_response(self, prompt):
        """Generates a response to a prompt using OpenAI's Davinci API"""
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.5,
            max_tokens=4000,
            n=1,
            stop=None
        )
        return response['choices'][0]['text']

    def transcribe(self, filename):
        """
        Transcribes audio to text using Whisper.
        Supported file formats: mp3, mp4, mpeg, mpga, m4a, wav, and webm.
        """
        try:
            result = self.model.transcribe(filename, fp16=self.fp16, language=None, condition_on_previous_text=False)
            return result["text"]
        except Exception as e:
            print("[Whisper] An error occurred: {}".format(e))
            # print("...")

    def translate_whisper(self):
        """Translates audio to English using OpenAI's Whisper API"""
        try:
            # Record audio
            audio_file = 'translate.wav'
            print("Recording...")
            with sr.Microphone() as source:
                recognizer = sr.Recognizer()
                source.pause_threshold = 1
                audio = recognizer.listen(source, phrase_time_limit=None, timeout=None)
                with open(audio_file, "wb") as f:
                    f.write(audio.get_wav_data())
                
            # Transcribe audio to text
            # transcript = openai.Audio.transcribe('whisper-1', audio_file)
            transcript = self.model.transcribe(audio_file, fp16=True, language=None, task='translate')
            if transcript:
                print(f"Transcription: {transcript}")
                
                # Generate Translation
                response = openai.Audio.translate('whisper-1', transcript)
                print(f"Translation: {response}")
                    
        except Exception as e:
            print("[Translate] An error occurred: {}".format(e))


""" Fast Whisper Model """
class FastWhisper:
    def __init__(self) -> None:
        pass


def main():
    # Example usage. Use supported file formats.
    wisp = Wisp()
    text = wisp.transcribe("output.mp3")
    print(text)

if __name__ == "__main__":
    main()