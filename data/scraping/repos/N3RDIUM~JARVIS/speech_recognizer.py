import speech_recognition as sr
import os
import openai

openai.api_base = "http://localhost:8080/v1"
openai.api_key = "sx-xxx"
OPENAI_API_KEY = "sx-xxx"
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

s = sr.Recognizer()
m = sr.Microphone()
with m as source:
    s.adjust_for_ambient_noise(source)
    
def listen_and_save():
    with m as source:
        audio = s.listen(source)
    with open(".cache/microphone-results.wav", "wb") as f:
        f.write(audio.get_wav_data())
    return audio

def transcribe(audio):
    return s.recognize_google(audio)

def transcribe_whisper():
    results = openai.Audio.transcribe(
        model="whisper-ggml-small.en.bin",
        file=open(".cache/microphone-results.wav", "rb")
    )
    return results["text"]