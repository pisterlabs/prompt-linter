import openai

from config.config import ENV_VARIABLES

openai.api_key = ENV_VARIABLES["OPENAI_API_KEY"]


class WhisperClass:
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def transcribe(self, audio_file):
        transcript = openai.Audio.transcribe(file=audio_file, model=self.model_name)
        return transcript['text']
