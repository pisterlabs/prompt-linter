import openai
import os


class WhisperSpeechToText:
    def __init__(self, openai_api_key):
        os.environ["OPENAI_API_KEY"] = openai_api_key

    def transcribe_audio(self, audio_file_path):
        with open(audio_file_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
        return transcript["text"]
