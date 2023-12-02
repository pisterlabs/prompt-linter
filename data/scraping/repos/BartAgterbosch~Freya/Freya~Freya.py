import os
import openai
import tempfile
from sys import exit as term
import speech_recognition as recognition
from pathlib import Path as path
from datetime import datetime
from gtts import gTTS
from googletrans import Translator
from playsound import playsound
from time import sleep

def get_key():
    if (path("freya_key.dat").exists()):
        key = path("freya_key.dat").read_text().lstrip().lstrip(" ").split(" ")[0].rstrip("\n")
        if (len(key) > 30):
            return key
        else:
            pwd = path("freya_key.dat").absolute()
            print(f"Invalid key or keyformat, please check {pwd} and make sure there is nothing in there except the key, and that the key itself is correct.")
            sleep(2)
            input("Press enter to exit now.")
            term("Invalid key.")
    else:
        path("freya_key.dat").touch()
        pwd = path("freya_key.dat").absolute()
        print(f"API keyfile not found, created file {pwd}, please enter your openai api key in the file.")
        sleep(2)
        input("Press enter to exit now.")
        term("File not found.")

speech  = recognition.Recognizer()
translate = Translator()
name = "Freya"
date = str(datetime.today()).split(" ")[0]
content = ""

def gpt3(stext):
    openai.api_key = get_key()
    response = openai.Completion.create(
        engine="text-davinci-001",
        prompt=stext,
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=["You: ", f"{name}: "]
    )
    return response.choices[0].text

content +=f"Your name is {name}\n"
response = gpt3(content).lstrip()
content += f"{response}\n"

try:
    while (True):
        while (True):
            with recognition.Microphone() as source:
                speech.energy_threshold = 4000
                print("Listening..")
                audio = speech.listen(source)
            try:
                text = speech.recognize_google(audio)
                break
            except:
                print("I did not catch that, let's try again.")
                pass
        user = f"You: {text}"
        print(user)
        content += f"{user}\n"
        response = gpt3(content).lstrip()
        content += f"{response}\n"
        autolang = str(translate.detect(response)).split("=")[1].split(",")[0]
        sentence = gTTS(text=response, slow=False, lang=autolang, tld='nl')
        mp3 = tempfile.NamedTemporaryFile(suffix=".mp3")
        mp3 = mp3.name
        try:
            os.remove(mp3)
        except:
            pass
        sleep(1)
        sentence.save(mp3)
        sleep(1)
        print(f"{name}: {response}\n")
        playsound(mp3)
        sleep(1)
        try:
            os.remove(mp3)
        except:
            pass
except KeyboardInterrupt:
    pass
