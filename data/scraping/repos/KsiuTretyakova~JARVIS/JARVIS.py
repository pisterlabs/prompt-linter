import webbrowser

import speech_recognition as sr
import os
import sys

# -------------------------------------------------
import openai

from dotenv import load_dotenv as ld
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(dotenv_path):
    ld(dotenv_path)

openai.api_key = os.getenv("api_key")

def handle_input(user_input):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": user_input}])
    return completion
# -----------------------------------------------------------

# Якщо немає звуку
# pip install pyttsx3
import pyttsx3
engine = pyttsx3.init()
# engine.say("Say")
# engine.runAndWait()

take = input("Який тип помычника ти хочеш? Обери голосовий (1) чи письмовий (2) ")

def talk(words):
    print(words)
    if take == "1":
        engine.say(words)
        engine.runAndWait()
        # os.system("say " + words)


talk("Hi, can I help you?")


def command():
    if take == "1":
        r = sr.Recognizer()

        # source = sr.Microphone()

        with sr.Microphone() as source:
            print("Say")
            r.pause_threshold = 1
            r.adjust_for_ambient_noise(source, duration=1)
            audio = r.listen(source)

        try:
            # task = r.recognize_google(audio, language="en-US").lower()
            task = r.recognize_google(audio, language="uk-UA").lower()
            print("Ви проговорили: " + task)
        except sr.UnknownValueError:
            # talk("Я вас не зрозумів")
            talk("Ya was ne zrozymiv")
            task = command()
    else:
        task = input("Your task: ")

    return task


def make_something(task):
    # if "open site" in task:
    if ("відкрий" and "сайт") in task:
        talk("Відкриваю")
        url = "https://ituniver.com"
        webbrowser.open(url)

    elif "ім'я" and "твоє" in task:
        talk("My name`s JARVIS")

    elif "стоп" in task:
        talk("Good buy")
        sys.exit()

    else:
        try:
            ai_response = handle_input(task).choices[0].message.content
            talk(ai_response)
        except openai.error.ServiceUnavailableError:
            talk("Sorry, I am going to try again")
            try:
                ai_response = handle_input(task).choices[0].message.content
                talk(ai_response)
            except openai.error.ServiceUnavailableError:
                talk("Sorry, can you give me the new task?")


while True:
    make_something(command())
