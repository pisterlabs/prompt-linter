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
import pywhatkit
from translate import Translator
from random import choice
from pprint import pprint
import matplotlib.pyplot as plt
import cv2
import subprocess
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


if __name__ == "__main__":
    greet_user()
    speak("Hi, I am Koeus, your personal virtual assistant. How can I help you?")

    while True:
        query = take_command().lower()
        home_user_dir = os.path.expanduser("~")

        if 'from wikipedia' in query:
            speak('Searching Wikipedia...')
            query = query.replace("wikipedia", "")
            results = wikipedia.summary(query, sentences=2)
            speak("According to Wikipedia")
            print(results)
            speak(results)

        elif 'from youtube' in query:
            speak('What do you want to play on Youtube, sir?')
            video = take_command().lower()
            video = video.replace(" ", "+")
            start = "https://www.youtube.com/results?search_query="
            final = "".join([start, video])
            webbrowser.open(final)

        elif 'search' in query:
            # Set up OpenAI API credentials
            openai.api_key = "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"
            query = query.replace("search", "")
            query = query.replace(" ", "+")
            starting_URL="https://www.google.com/search?q="
            ending_URL="&rlz=1C1CHBD_enIN1056IN1056&sxsrf=APwXEde9Saq_KqhU0sg8AzWDG7XXln4RwQ%3A1686055805145&ei=fSt_ZPiaCOPsseMP7s2V0Aw&ved=0ahUKEwi4zdiq167_AhVjdmwGHe5mBcoQ4dUDCBA&uact=5&oq=computer&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAzIHCCMQigUQJzIHCCMQigUQJzILCAAQgAQQsQMQgwEyCwgAEIAEELEDEIMBMg0ILhCKBRDHARDRAxBDMgsIABCABBCxAxCDATIICAAQgAQQyQMyCAgAEIoFEJIDMggIABCKBRCSAzILCAAQgAQQsQMQgwE6BwgjELADECc6CggAEEcQ1gQQsAM6DAgAEIoFELADEAoQQzoKCAAQigUQsAMQQzoNCAAQ5AIQ1gQQsAMYAToVCC4QigUQxwEQ0QMQyAMQsAMQQxgCOg8ILhCKBRDIAxCwAxBDGAJKBAhBGABQ1gVY1gVg8QtoAXABeACAAaMBiAGjAZIBAzAuMZgBAKABAcABAcgBEdoBBggBEAEYCdoBBggCEAEYCA&sclient=gws-wiz-serp"
            URL= "".join([starting_URL, query, ending_URL])
            webbrowser.open(URL)

        elif 'open google' in query:
            webbrowser.open("https://www.google.com/")

        elif 'open github' in query:
            webbrowser.open("https://github.com/Ignius-Namikaze")

        elif 'play' in query:
            query = query.replace("play", "")
            os.system("spotify")
            time.sleep(5)
            pyautogui.hotkey('ctrl', 'l')
            pyautogui.write(query, interval=0.5)

            for key in ['enter', 'pagedown', 'tab', 'enter', 'enter']:
                time.sleep(2)
                pyautogui.press(key)    

        elif 'what is the time' in query:
            strTime = datetime.datetime.now().strftime("%H:%M:%S")
            speak(f"The time is {strTime}")

        elif 'what is the date' in query:
            strDate = datetime.datetime.today().strftime('%Y-%m-%d')
            print(strDate)
            speak(f"The date is {strDate}")

        elif 'open whatsapp' in query:
            os.startfile(home_user_dir + "\\AppData\\Local\\WhatsApp\\WhatsApp.exe")

        elif 'who are you' in query:
            speak("I am Koeus, a virtual assistant developed by Tungishsanjay Sankar ")

        elif 'what you want to do' in query:
            speak("I want to help people to do certain tasks on their single voice commands.")

        elif 'alexa' in query:
            speak("She was my classmate, was dumb person. We both are best friends.")

        elif 'google assistant' in query:
            speak("He was my classmate, too intelligent guy. We both are best friends.")

        elif 'siri' in query:
            speak("Siri, She's a competing virtual assistant on a competitor's phone. "
                        "Not that I'm competitive or anything.")

        elif 'cortana' in query:
            speak("I thought you'd never ask. So I've never thought about it.")

        elif 'python assistant' in query:
            speak("Are you joking. You're coming in loud and clear.")

        elif 'what language you use' in query:
            speak("I am written in Python and I generally speak english.")

        elif 'price of' in query:
            query = query.replace('price of', '')
            query = "https://www.amazon.in/s?k=" + query #indexing since I only want the keyword
            webbrowser.open(query)

        elif 'resume' in query or 'pause' in query:
            pyautogui.press("playpause")

        elif 'previous' in query:
            pyautogui.press("prevtrack")

        elif 'next' in query:
            pyautogui.press("nexttrack")

        elif 'set alarm' in query:
            set_alarm()

        elif 'weather' in query or 'temperature' in query:
            api_key = "e75d3629ef4caa4b70d11007f860b3e7"
            base_url = "https://api.openweathermap.org/data/2.5/weather?"
            speak("whats the city name")
            city_name = take_command()
            complete_url = base_url+"appid="+api_key+"&q="+city_name
            response = requests.get(complete_url)
            x = response.json()
            if x["cod"] != "404":
                y = x["main"]
                current_temperature = y["temp"]
                current_humidiy = y["humidity"]
                z = x["weather"]
                weather_description = z[0]["description"]
                speak(" Temperature in kelvin unit is " +
                      str(current_temperature) +
                      "\n humidity in percentage is " +
                      str(current_humidiy) +
                      "\n description  " +
                      str(weather_description))

            else:
                speak(" City Not Found ")

        elif 'month' in query or 'month is going' in query:
            def tell_month():
                month = datetime.datetime.now().strftime("%B")
                speak(month)

            tell_month()

        elif 'day' in query or 'day today' in query:
            def tell_day():
                day = datetime.datetime.now().strftime("%A")
                speak(day)

            tell_day()

        elif "calculate" in query:
            try:
                app_id = "JUGV8R-RXJ4RP7HAG"
                client = wolframalpha.Client(app_id)
                indx = query.lower().split().index('calculate')
                query = query.split()[indx + 1:]
                res = client.query(' '.join(query))
                answer = next(res.results).text
                print("The answer is " + answer)
                speak("The answer is " + answer)

            except Exception as e:
                print("Couldn't get what you have said, Can you say it again??")

        elif 'quote' in query or 'quotes' in query:
            speak("Tell me the author or person name.")
            q_author = take_command()
            quotes = quote(q_author)
            quote_no = random.randint(1, len(quotes))
            # print(len(quotes))
            # print(quotes)
            print("Author: ", quotes[quote_no]['author'])
            print("-->", quotes[quote_no]['quote'])
            speak(f"Author: {quotes[quote_no]['author']}")
            speak(f"He said {quotes[quote_no]['quote']}")

        elif 'write a note' in query or 'make a note' in query:
            speak("What should I write, sir??")
            note = take_command()
            file = open('Notes.txt', 'a')
            speak("Should I include the date and time??")
            n_conf = take_command()
            if 'yes' in n_conf:
                str_time = datetime.datetime.now().strftime("%H:%M:%S")
                file.write(str_time)
                file.write(" --> ")
                file.write(note)
                speak("Point noted successfully.")
            else:
                file.write("\n")
                file.write(note)
                speak("Point noted successfully.")

        elif 'show me the notes' in query or 'read notes' in query:
            speak("Reading Notes")
            file = open("Notes.txt", "r")
            data_note = file.readlines()
            # for points in data_note:
            print(data_note)
            speak(data_note)

        elif 'distance' in query:
            geocoder = Nominatim(user_agent="Singh")
            speak("Tell me the first city name??")
            location1 = take_command()
            speak("Tell me the second city name??")
            location2 = take_command()

            coordinates1 = geocoder.geocode(location1)
            coordinates2 = geocoder.geocode(location2)

            lat1, long1 = coordinates1.latitude, coordinates1.longitude
            lat2, long2 = coordinates2.latitude, coordinates2.longitude

            place1 = (lat1, long1)
            place2 = (lat2, long2)

            distance_places = distance.distance(place1, place2)

            print(f"The distance between {location1} and {location2} is {distance_places}.")
            speak(f"The distance between {location1} and {location2} is {distance_places}")

        elif 'screenshot' in query:
            sc = pyautogui.screenshot()
            sc.save('pa_ss.png')
            print("Screenshot taken successfully.")
            speak("Screenshot taken successfully.")

        elif 'volume up' in query:
            pyautogui.press("volumeup")

        elif 'volume down' in query:
            pyautogui.press("volumedown")

        elif 'mute volume' in query:
            pyautogui.press("volumemute")

        elif 'shut down' in query:
            print("Do you want to shutdown you system?")
            speak("Do you want to shutdown you system?")
            cmd = take_command()
            if 'no' in cmd:
                continue
            else:
                
                os.system("shutdown /s /t 1")

        elif 'restart' in query:
            print("Do you want to restart your system?")
            speak("Do you want to restart your system?")
            cmd = take_command()
            if 'no' in cmd:
                continue
            else:
                
                os.system("shutdown /r /t 1")

        elif 'log out' in query:
            print("Do you want to logout from your system?")
            speak("Do you want to logout from your system?")
            cmd = take_command()
            if 'no' in cmd:
                continue
            else:
                os.system("shutdown -l")

        elif 'send message on whatsapp' in query:
            speak("Whom do you want to send message to")
            name = take_command()
            speak("What message do you want to send")
            msg = take_command()
            webbrowser.open("https://web.whatsapp.com/")
            time.sleep(52)
            for key in ['tab', 'tab', 'tab', 'tab', 'tab']:
                time.sleep(0.5)
                pyautogui.press(key)
            pyautogui.write(name, interval=0.5)

            for key in ['enter']:
                time.sleep(2)
                pyautogui.press(key)
            pyautogui.write(msg, interval=0.8)

            for key in ['enter']:
                time.sleep(1)
                pyautogui.press(key)

        elif 'news' in query or 'news headlines' in query:
            speak("Which news do you want")
            type = take_command()
            new_url = 'https://www.bbc.com/news/'
            url = "".join([new_url, type])
            response = requests.get(url)

            soup = BeautifulSoup(response.text, 'html.parser')
            headlines = soup.find_all('h3')

            for headline in headlines:
                speak(headline.text.strip())

        elif 'joke' in query or 'jokes' in query or 'tell me a joke' in query:

            joke = pyjokes.get_joke()
            speak(joke)
            print(joke)

        elif 'set a reminder' in query:
            # set up the credentials
            creds = service_account.Credentials.from_service_account_file(
                "D:\Personal Assistant\koeus-392406-e9cb4dc089db.json",
            scopes=['https://www.googleapis.com/auth/calendar'])

            # create the calendar service
            service = build('calendar', 'v3', credentials=creds)

            speak("What should be the start time?")
            speak("The format should be [Year, Month, Day, 24Hour, Mins, Seconds]")
            st = take_command()
            # set the start and end times for the event
            start_time = datetime(st)
            end_time = start_time + timedelta(hours=1)

            speak("What is the reminder about?")
            Su = take_command()
            speak("Can you describe the event?")
            De = take_command()
            # create the event
            event = {
                    'summary': Su,
                    'description': De,
                    'start': {
                    'dateTime': start_time.isoformat(),
                    'timeZone': 'India/Delhi',
                        },
                    'end': {
                    'dateTime': end_time.isoformat(),
                    'timeZone': 'India/Delhi',
                        },
                    'reminders': {
                    'useDefault': True,
                        },
                    }

            # insert the event into the calendar
            event = service.events().insert(calendarId='primary', body=event).execute()
            print('Event created: %s' % (event.get('htmlLink')))
            speak("Event Created")
        elif 'goodbye' in query:
            speak("Goodbye! Have a")
            greet_user()
            break
