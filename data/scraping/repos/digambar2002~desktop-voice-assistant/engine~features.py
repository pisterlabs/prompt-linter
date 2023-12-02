import re
from engine.config import *
from unittest import result
import requests
from bs4 import BeautifulSoup
import json
from ast import dump
from audioop import avgpp
from sys import path
import sys
import eel
from pyparsing import Empty
import requests

# This Module is ued to convert text to speech
import pyttsx3

# This module is used to recoginize speech command
import speech_recognition as sr

# Date Time Module to get current date and time
import datetime

# Wkipedia Module to search things on wikipedia
import wikipedia

# OS Module To work On Windows Like Open notepad or cmd
import os

# This Module is use to get time
import time

# This Module is used to play sounds and music
from playsound import playsound

# This module is used to open web browser
import webbrowser

# Give Randoms Facts
# import randfacts

# This function is used to send message or search on google
import pywhatkit as kit

# this module is used to automate system or uses keyboaed keys and mouse
import pyautogui as autogui


# python script showing battery details
import psutil

import sqlite3


# Global Declaration
connection = sqlite3.connect('assistant.sqlite')
cursor = connection.cursor()


engine = pyttsx3.init()
voices = engine.getProperty('voices')
# print(voices[1].id)
engine.setProperty('voice', voices[0].id)
engine.setProperty('rate', 174)

# text to speech


def speak(audio):
    engine.say(audio)
    eel.WishMessage(audio)
    eel.SpeakMessage(audio)
    eel.receiverText(audio)
    engine.runAndWait()
    return audio


# Main words function
def remove_words(word_list, string):
    pattern = r'\b(?:%s)\b' % '|'.join(word_list)
    modified_string = re.sub(pattern, ' ', string, flags=re.IGNORECASE)
    return modified_string


def auth_protocol():
    # Hide the loader screen and display face auth using js
    eel.hideLoader()
    speak('ready for face authentication')
    speak('performing face authentication')

# Battery status function


def battery():
    # function returning time in hh:mm:ss
    def convertTime(seconds):
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return "%d:%02d:%02d" % (hours, minutes, seconds)

    battery = psutil.sensors_battery()
    speak("Sir, Your Battery Status is: ")
    print("Battery percentage : ", battery.percent)
    speak(f"Battery Percentage: {battery.percent}")

    if battery.power_plugged == True:

        print("Power plugged in : ON")
        speak("Power plugged in is on")

    else:
        print("Power plugged in : OFF")
        speak("Power plugged in is OFF")

    batteryRemaning = str(convertTime(battery.secsleft))
    batterylist = batteryRemaning.split(':')

    speak(
        f"Battery left: {batterylist[0]} hours, {batterylist[1]} minutes, {batterylist[2]} seconds")
    print("Battery left : ", convertTime(battery.secsleft))

# Loading Effect


def loading():
    music_dir = "audio\\alert sound\\bell_alert.wav"
    speak("System Initiating")
    playsound(music_dir)
    speak("Initializing Database")
    eel.TextSet("Initializing Database")
    playsound(music_dir)
    speak("Adding All The Preferances")
    eel.TextSet("Adding All The Preferances")
    playsound(music_dir)
    speak("System is now fully operational")
    eel.TextSet("Starting ...")


# Time Whiches function
def wish():

    hour = int(datetime.datetime.now().hour)
    currentTime = datetime.datetime.now()
    currentTime = currentTime.strftime(
        '%I %M %p').lstrip("0").replace(" 0", " ")
    if hour > 0 and hour < 12:
        eel.WishMessage(speak("Hello, Good Morning "+OWNER_NAME))

    elif hour >= 12 and hour < 18:
        eel.WishMessage(speak("Hello, Good Afternoon "+OWNER_NAME))
    else:
        eel.WishMessage(speak("Hello, Good Evening "+OWNER_NAME))
    
    speak('How can i help you')

# Open Commands


def openCommand(query):
    query = query.replace(ASSISTANT_NAME, "")
    query = query.replace("open", "")
    query.lower()
    
    # System Command
    if query in query:
        cursor.execute(
            "SELECT path FROM sys_command WHERE name='%s'" % query.strip())
        results = cursor.fetchall()
        if len(results) != 0:
            flag = results[0]
            path = flag[0]
            repr(path)
            speak("Opening "+query)
            os.startfile(path)
        else:
            cursor.execute(
                "SELECT path FROM web_command WHERE name='%s'" % query.strip())
            results = cursor.fetchall()
            if len(results) != 0:
                flag = results[0]
                path = flag[0]
                repr(path)
                speak("Opening "+query)
                webbrowser.open(path)
            else:
                
                try:
                    os.system('start '+query)
                except:
                    speak("not found")
    else:
        pass


# openCommand(" notepad")


def close(query):
    query = query.replace(ASSISTANT_NAME, "")
    query = query.replace("close", "")

    if "notepad" in query:
        speak("Closing Notepad")
        os.system("TASKKILL /F /IM notepad.exe")

    elif "chrome" in query:
        speak("closing chrome")
        os.system("TASKKILL /F /IM chrome.exe")

    elif "xampp" in query:
        speak("closing xampp")
        os.system("TASKKILL /F /IM xampp-control.exe")

    elif "spotify" in query:
        speak("closing spotify")
        os.system("TASKKILL /F /IM spotify.exe")

    else:
        pass

# search on web browser


def searchTerm(query):
    query = query.replace(ASSISTANT_NAME, "")
    query = query.replace("search", "")
    speak("please sir, wait for a minute")

    kit.search(query)
    speak("here what i found on web")
    # term = wikipedia.summary(query, sentences=2)
    # speak(term)

# Chat Gpt

# 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


def chatGPT(query):
    query = query.replace(ASSISTANT_NAME, "")
    query = query.replace("search", "")

    import openai
    openai.api_key = "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"
    prompt = query

    try:

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=600,
            n=1,
            stop=None,
            temperature=0.5,
        )
        speak(response.choices[0].text.strip())
    except:
        speak("something went wrong")


def chatGPT2(query):
    query = query.replace(ASSISTANT_NAME, "")
    query = query.replace("search", "")
    prompt = query

    try:

        url = "https://open-ai21.p.rapidapi.com/chat"

        payload = {"message": prompt}
        headers = {
            "content-type": "application/json",
            "X-RapidAPI-Key": "d68c7a4ca0mshfe7db0559a72ad6p1f118fjsnb3beeaa77aac",
            "X-RapidAPI-Host": "open-ai21.p.rapidapi.com"
        }

        response = requests.post(url, json=payload, headers=headers)
        message = response.json()['ChatGPT']
        print(message)
        speak(message)
    except:
        speak("something went wrong")


# Play On YouTube


def PlayYoutube(query):
    query = query.replace(ASSISTANT_NAME, "")
    query = query.replace("search", "")
    query = query.replace("play", "")
    query = query.replace("on youtube", "")
    speak("Playing"+query+"on YouTube")
    kit.playonyt(query)

# Random Facts


# def RandomFacts():
#     fact = randfacts.getFact()
#     speak(fact)
#     print(fact)


# minimize all open window
def MinimizeOpenWindows():
    autogui.keyDown("win")
    autogui.press("d")
    time.sleep(2)
    autogui.keyUp("win")

# maximize all open window


def MaximizeOpenWindows():
    autogui.keyDown("win")
    autogui.press("d")
    time.sleep(2)
    autogui.keyUp("win")


def copy():
    autogui.hotkey('ctrl', 'c')


def paste():
    autogui.hotkey('ctrl', 'v')


#  ************************************************** WEATHER METHOD **********************************************

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}


def weather(query):
    query = query.replace("weather", "")
    query = query.replace("of", "")
    query = query.replace("in", "")
    print(query)
    if query == "":
        city = CITY_NAME + " weather"
    else:
        city = query+" weather"

    try:

        url = "https://weatherapi-com.p.rapidapi.com/current.json"

        querystring = {"q": city}

        headers = {
            "X-RapidAPI-Key": "d68c7a4ca0mshfe7db0559a72ad6p1f118fjsnb3beeaa77aac",
            "X-RapidAPI-Host": "weatherapi-com.p.rapidapi.com"
        }

        response = requests.get(url, headers=headers, params=querystring)

        weather = response.json()['current']['feelslike_c']
        info = response.json()['current']['wind_kph']

        eel.weatherShow(info, str(weather) +" °C", city, time)
        speak("its "+str(weather)+" degree celsius and wind speed is "+str(weather)+" kilometer per hour in "+city)
    except IndexError:
        speak("Can't found city " + query)


#  ************************************************** WEATHER METHOD Ends **********************************************

# Whatsapp Message Sending
def sendMessage(query):
    query = query.replace(ASSISTANT_NAME, "")
    query = query.replace("send", "")
    query = query.replace("message", "")
    query = query.replace("to", "")
    query = query.replace("wahtsapp", "")
    try:
        cursor.execute(
            "SELECT mobileno FROM phonebook WHERE name='%s'" % query.strip().lower())
        results = cursor.fetchall()
        return results[0][0]
    except:
        speak('not exist in contacts')
        return 0


def whatsAppSend(mobile_no, message):
    current_time = datetime.datetime.now()
    current_hour = current_time.hour
    current_minute = current_time.minute
    try:
        speak("sending message ....")
        kit.sendwhatmsg(mobile_no, message, current_hour, current_minute+1)
        speak("message sent successfully")

    except:
        speak("something went wrong")


# Make Phone Call Command
def MakeCall(query):
    query = query.replace(ASSISTANT_NAME, "")
    query = query.replace("to", "")
    query = query.replace("make a", "")
    query = query.replace("phone", "")
    query = query.replace("call", "")
    print(query.strip())
    cursor.execute(
        "SELECT mobileno FROM phonebook WHERE name='%s'" % query.strip().lower())
    results = cursor.fetchall()
    if len(results) != 0:
        speak("Calling "+query)
        flag = results[0]
        MobileNo = flag[0]
        command = 'adb shell am start -a android.intent.action.CALL -d tel:+91'+MobileNo
        os.system(command)
    else:
        speak('No Data Found')


def DisconnectCall():
    command = 'adb shell service call phone 5'
    speak("disconnecting call...")
    os.system(command)


# Settings Function
def systemCommand():
    cursor.execute("SELECT * FROM sys_command")
    results = cursor.fetchall()
    print(results)


# Music Player

def spotifyPlayer(query):
    word_list = [ASSISTANT_NAME, 'play', 'music', 'spotify', 'to', 'song']

    songName = remove_words(word_list, query)

    url = "https://spotify23.p.rapidapi.com/search/"

    querystring = {"q": songName, "type": "tracks", "offset": "0",
                   "limit": "10", "numberOfTopResults": "5"}

    headers = {
        "X-RapidAPI-Key": "d68c7a4ca0mshfe7db0559a72ad6p1f118fjsnb3beeaa77aac",
        "X-RapidAPI-Host": "spotify23.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    items = response.json()['tracks']['items']

    song = ""
    for x in items:
        song = x['data']['id']
        break

    print(song)
    speak("Playing"+songName+" on Spotify")
    command = "start spotify:track:"+song
    os.system(command)


# Assistant name
@eel.expose
def assistantName():
    name = ASSISTANT_NAME
    return name


@eel.expose
def personalInfo():
    cursor.execute("SELECT * FROM info")
    results = cursor.fetchall()
    jsonArr = json.dumps(results[0])
    eel.getData(jsonArr)
    return 1


@eel.expose
def updatePersonalInfo(name, desiganation, mobileno, email, city):
    cursor.execute('''UPDATE info SET name=?, designation=?, mobileno=?, email=?, city=? ''',
                   (name, desiganation, mobileno, email, city))
    connection.commit()
    personalInfo()
    return 1


@eel.expose
def displaySysCommand():
    cursor.execute("SELECT * FROM sys_command")
    results = cursor.fetchall()
    jsonArr = json.dumps(results)
    eel.displaySysCommand(jsonArr)
    return 1


@eel.expose
def deleteSysCommand(id):
    cursor.execute(
        ''' DELETE FROM sys_command WHERE name= '%s' ''' % id.strip())
    connection.commit()


@eel.expose
def addSysCommand(key, value):
    cursor.execute(
        '''INSERT INTO sys_command VALUES (?, ?)''', (key, value))
    connection.commit()


@eel.expose
def displayWebCommand():
    cursor.execute("SELECT * FROM web_command")
    results = cursor.fetchall()
    jsonArr = json.dumps(results)
    eel.displayWebCommand(jsonArr)
    return 1


@eel.expose
def addWebCommand(key, value):
    cursor.execute(
        '''INSERT INTO web_command VALUES (?, ?)''', (key, value))
    connection.commit()


@eel.expose
def deleteWebCommand(id):
    cursor.execute(
        ''' DELETE FROM web_command WHERE name= '%s' ''' % id.strip())
    connection.commit()


@eel.expose
def displayPhoneBookCommand():
    cursor.execute("SELECT * FROM phonebook")
    results = cursor.fetchall()
    jsonArr = json.dumps(results)
    eel.displayPhoneBookCommand(jsonArr)
    return 1


@eel.expose
def deletePhoneBookCommand(id):
    cursor.execute(
        ''' DELETE FROM phonebook WHERE mobileno= '%s' ''' % id.strip())
    connection.commit()


@eel.expose
def InsertContacts(Name, MobileNo, Email, City):
    cursor.execute(
        '''INSERT INTO phonebook VALUES (?, ?, ?, ?)''', (Name, MobileNo, Email, City))
    connection.commit()
