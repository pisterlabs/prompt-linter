import os
import time
import pyaudio
import playsound
from gtts import gTTS
import openai
import speech_recognition as sr

api_key = "--Your__API__KEY"
openai.api_key = api_key

lang = 'en'

said = ""

def get_audio():
    r = sr.Recognizer()
    with sr.Microphone(device_index=0) as source:
        audio = r.listen(source)
        said = ""

def get_response():
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=[{"role":  "user", "content": said}])
    text = completion.choices[0].message.content
    speech=gTTS(text=text, lang=lang, slow=False, tld="com.au")
    speech.save("welcome1.mp3")
    playsound.playsound("welcome1.mp3")

get_audio()
get_response()


				
        
        
            

