import dotenv
dotenv.load_dotenv()
import os
import openai

def convert_to_text(audio_data):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    transcript = openai.Audio.transcribe("whisper-1", audio_data)
    return transcript["text"]
    
def test_convert_to_text():
    audio_file = open("output.m4a", "rb")
    print(convert_to_text(audio_file))
