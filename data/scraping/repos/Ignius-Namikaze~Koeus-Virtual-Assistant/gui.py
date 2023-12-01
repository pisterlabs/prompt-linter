from subprocess import call
import sys
import pyttsx3
import speech_recognition as sr
import wikipedia
import webbrowser
import datetime
import openai
import os
import time
import pyautogui
import pyjokes
import ecapture as ec
import speedtest
import poetpy
from bs4 import BeautifulSoup
import wolframalpha
from quote import quote
from geopy.geocoders import Nominatim
from geopy import distance
from requests_html import HTMLSession
import requests
from PIL import Image
import random
from translate import Translator
from random import choice
from pprint import pprint
import matplotlib.pyplot as plt
import cv2
from playsound import playsound
from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import timedelta

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[2].id) 

def speak(text):
    engine.say(text)
    engine.runAndWait()

def take_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-in').lower()
        print(f"User said: {query}\n")
    except Exception as e:
        print("Say that again please...")
        return "None"
    return query

def greet_user():
    """Greets the user according to the time"""
    hour = datetime.datetime.now().hour
    if (hour >= 6) and (hour < 12):
        speak("Good Morning!")
    elif (hour >= 12) and (hour < 16):
        speak("Good afternoon!")
    elif (hour >= 16) and (hour < 19):
        speak("Good Evening!")
    else:
        speak("Good Night!")

def set_alarm():
    speak("Enter the alarm time (24 hrs format): ")
    alarm_time = take_command().lower()
    alarm_time = alarm_time.replace(" hours ",":")
    alarm_time = alarm_time.replace(" minutes","")
    while True:
        current_time = datetime.datetime.now().strftime("%H:%M")
        difference = (datetime.datetime.strptime(alarm_time, "%H:%M") - datetime.datetime.strptime(current_time, "%H:%M")).total_seconds()
        if difference < 0:
            difference += 86400  
        if difference == 0:
            print("Wake up!")
            playsound("D:\Personal Assistant\Alarm_Rigntone.mp3")  
            break
        time.sleep(1)


from pathlib import Path

from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"D:\Personal Assistant\build\assets\frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def run_script():
    global var
    if var.get() == 0:
        var.set(1)
        exec(open("koeus.py").read())
        button_image_1 = PhotoImage(
    file=relative_to_assets("button_0.png"))
    else:
        var.set(0)
        button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
        window.quit()
    

window = Tk()

window.geometry("466x61")
window.configure(bg = "#A0A0A0")
window.overrideredirect(True)

var = BooleanVar()
var.set(False)

canvas = Canvas(
    window,
    bg = "#A0A0A0",
    height = 61,
    width = 466,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: run_script(),
    relief="flat"
)
button_1.place(
    x=0.0,
    y=4.0,
    width=57.0,
    height=53.0
)

entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    259.0,
    30.5,
    image=entry_image_1
)
label_1 = Label(text="Koeus"
    #bd=0,
    #bg="#FFFFFF",
    #fg="#000716",
    #highlightthickness=0
)
label_1.place(
    x=57.0,
    y=4.0,
    width=404.0,
    height=51.0
)
window.resizable(False, False)
window.mainloop()
