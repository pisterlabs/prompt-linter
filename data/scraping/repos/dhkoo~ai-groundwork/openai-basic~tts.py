from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class SpeechGenerator:
    def __init__(self, api_key: str, model: str = 'tts-1', voice: str = 'nova'):
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
    api_key = os.getenv("OPENAI_API_KEY")
    speech_gen = SpeechGenerator(api_key)

    text = "안녕하세요! 저는 에이릿입니다. 만나서 반갑습니다."
    speech_file_path = "output.mp3"

    speech_gen.generate_speech(text, speech_file_path)

if __name__ == "__main__":
    main()