from elevenlabs import play
import openai
import speech_recognition as sr
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")

r = sr.Recognizer()
mic = sr.Microphone(device_index=0)

with mic as source:
    print("请讲话...")
    r.adjust_for_ambient_noise(source)
    audio_data = r.record(source, duration=5)


with open("audio.wav", "wb") as f:
    f.write(audio_data.get_wav_data())

with open("audio.wav", "rb") as audio_file:
    transcript = openai.Audio.transcribe(
        "whisper-1", audio_file, language="zh")

text = transcript.text

print(text)

play(open("audio.wav", "rb").read())