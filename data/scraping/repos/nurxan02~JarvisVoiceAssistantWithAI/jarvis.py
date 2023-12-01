
from openai import OpenAI

client = OpenAI(api_key='YOUR_API_KEY')
import speech_recognition as sr
from gtts import gTTS
import os



def process_command(command):
    response = client.completions.create(model="text-davinci-003",
    prompt=command,
    max_tokens=100)
    return response.choices[0].text.strip()

def speak(text):
    tts = gTTS(text=text, lang='tr')
    tts.save("response.mp3")
    os.system("mpg321 response.mp3")

def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Dinleniyor...")
        audio = recognizer.listen(source)
    try:
        print("Tanınıyor...")
        command = recognizer.recognize_google(audio, language="tr-TR")
        print("Söylediğiniz: " + command)
        return command
    except sr.UnknownValueError:
        print("Ses anlaşılamadı.")
        return ""
    except sr.RequestError as e:
        print("Sonuçlar alınamadı: {0}".format(e))
        return ""

while True:
    command = listen()
    if "çıkış" in command:
        break
    response = process_command(command)
    speak(response)
