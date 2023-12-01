import pyttsx3  # pip install pyttsx3
import datetime  # module
import speech_recognition as sr
import openai
import wikipedia
import smtplib
import webbrowser
import os  # inbuilt
import pyautogui
import psutil  # pip install psutil
import pyjokes  # pip install pyjokes
import requests
import json  # inbuilt
import random
import threading
import twilio
import pywhatkit
import calendar
from playsound import playsound

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.setProperty('rate', 150)

# t=datetime.datetime.today()g


def voice_change(v):
    x = int(v)
    engine.setProperty('voice', voices[x].id)
    speak("done sir")


def speak(command):
    engine.say(command)
    engine.runAndWait()


def cpu():
    usage = str(psutil.cpu_percent())
    speak('CPU usage is at ' + usage)
    print('CPU usage is at ' + usage)
    battery = psutil.sensors_battery()
    speak("Battery is at")
    speak(battery.percent)
    print("battery is at:" + str(battery.percent))


def chat(message):
    openai.api_key = "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"
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=message,
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )
        reply = response.choices[0]['text']
        print(reply)
        if len(reply)<50:
            speak(reply)
    except:
        print('try again')


def jokes():
    j = pyjokes.get_joke()
    print(j)
    speak(j)


def weather():
    city = input("Enter the city name: ")
    url = "http://api.openweathermap.org/data/2.5/weather?q=" + city + "&appid=d8f8f2c9a9e9b7d0a7a7a9c9b8a5a0c3"
    response = requests.get(url)
    data = response.json()
    print(data)
    temp = data['main']['temp']
    print(temp)
    temp = int(temp - 273.15)
    print(temp)
    print("The temperature in " + city + " is " + str(temp) + " degree celsius")
    speak("The temperature in " + city + " is " + str(temp) + " degree celsius")   


def takecommand():
    s = sr.Recognizer()
    with sr.Microphone() as source:
        print("speak now")
        s.pause_threshold = 1
        s.energy_threshold = 1000
        audio = s.listen(source)
    try:
        print("recognizing....")
        query = s.recognize_google(audio, language="en-in")
        print(query)
    except:
        speak("couldn't recognize your voice")
        return 'none'
    return query


def username():
    speak("what's your name")
    name = takecommand()
    speak(f"hello {name} how can i help you")


def wish():
    hr = int(datetime.datetime.now().hour)
    if hr >= 0 and hr < 12:
        speak("good morning")
    elif hr >= 12 and hr < 18:
        speak("good afternoon")
    else:
        speak("good evening")

    intro()
    #username()



def intro():
    speak("hello i am jarvis ")


if __name__ == "__main__":
    wish()

    while True:
        query = takecommand().lower()
        # if 'who are you' in query:
        #     speak("i am jarvis")

        # elif "who made you" in query or "who created you" in query:
        #     speak("I have been created by a")

        if 'open youtube' in query:
            webbrowser.open("https://www.youtube.com/")

        elif 'open google' in query:
            webbrowser.open("https://www.google.com/")

        elif ('wikipedia' in query or 'what' in query
              or 'when' in query or 'where' in query):
            speak("searching...")
            query = query.replace("wikipedia", "")
            query = query.replace("search", "")
            query = query.replace("what", "")
            query = query.replace("when", "")
            query = query.replace("where", "")
            query = query.replace("who", "")
            query = query.replace("is", "")
            result = wikipedia.summary(query, sentences=2)
            print(query)
            print(result)
            speak(result)

        elif 'trending video' in query:
            pywhatkit.playonyt("https://www.youtube.com/feed/trending")

        elif 'video in youtube' in query:
            query = query.replace('video in youtube', '')
            pywhatkit.playonyt(query)

        elif 'song in youtube' in query:
            query = query.replace('play song in youtube', '')
            pywhatkit.playonyt(query)

        elif 'trending song in youtube' in query:
            pywhatkit.playonyt(
                'https://www.youtube.com/playlist?list=PLFgquLnL59alwzM3kQ6cWhwwZXa4TCjN4')

        # elif 'time' in query:
        #     time = datetime.datetime.now().strftime('%I %M %p')
        #     print(f"the time is {time}")
        #     speak(f"the time is {time}")

        # elif 'date' in query:
        #     date = datetime.datetime.now().strftime('%d %B %Y')
        #     print(f"date is {date}")
        #     speak(f"date is {date}")

        # elif 'today day' in query:
        #     day = datetime.datetime.now().weekday()
        #     print(calendar.day_name[day])
        #     speak(calendar.day_name[day])

        

        # elif 'how are you' in query:
        #     speak("i am fine")
        #     speak("how are you ")

        # elif 'fine' in query or 'good' in query:
        #     speak(f"it's good to know that you are {query}")

        elif ("logout" in query):
            os.system("shutdown -1")
        elif ("restart" in query):
            os.system("shutdown /r /t 1")
        elif ("shut down" in query):
            os.system("shutdown /r /t 1")

        elif ("create a reminder list" in query or "reminder" in query):
            speak("What is the reminder?")
            data = takecommand()
            speak("You said to remember that" + data)
            reminder_file = open("data.txt", 'a')
            reminder_file.write('\n')
            reminder_file.write(data)
            reminder_file.close()

        elif ("cpu and battery" in query or "battery" in query or "cpu" in query):
            cpu()

        elif ("tell me a joke" in query or "joke" in query):
            jokes()

        elif ("weather" in query or "temperature" in query):
            weather()

        elif ("tell me your powers" in query or "help" in query
        or "features" in query):
            features = ''' i can help to do lot many things like..
            i can tell you the current time and date,
            i can tell you the current weather,
            i can tell you battery and cpu usage,
            i can create the reminder list,
            i can send email to anyone,
            i can shut down or logout or hibernate your system,
            i can tell you jokes,
            i can open any website,
            i can search the thing on wikipedia,
            i can change my voice from david to zira and vice-versa
            And yes one more thing, My team is working on this system to add more features...,
            tell me what can i do for you??
            '''
            print(features)
            speak(features)

        elif ("voice" in query):
            if voices[1]:
                voice_change(0)
            elif voices[0]:
                voice_change(1)

        # elif "will you be my GF" in query or "will you be my bf" in query:
        #     speak("what")

        # elif 'how are you' in query:
        #     speak("I am fine, Thank you")
            
        # elif 'fine' in query or "good" in query:
        #     speak("It's good to know that your fine")


        # elif "i love you" in query:
        #     speak("ok")

        # elif "who i am" in query:
        #     speak("If you talk then definately you are human.")


        # elif "who made you" in query or "who created you" in query:
        #     speak("I have been created by a final year student of BBDNIIT")


        elif 'quit' in query or 'exit' in query or 'bye' in query:
            speak("signing off")
            break

        else:
            chat(query)