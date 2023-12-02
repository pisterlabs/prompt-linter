import win32com.client
import speech_recognition as sr
import os
import webbrowser
import openai
import subprocess, sys


def say(text):
   os.system(f"say {text}")

def takecommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        speaker = win32com.client.Dispatch("SAPI.SpVoice") # create speaker object
        r.pause_threshold =  1
        audio = r.listen(source)

        try:
            print("Recognizing...")
            query = r.recognize_google(audio, language="en-in")
            print(f" User said: {query}")
            return query
        except Exception as e:
            return "some error occured"

if __name__ == '__main__':
    print('hello')
    speaker = win32com.client.Dispatch("SAPI.SpVoice") # create speaker object
    speaker.speak("hello YUVRAJ , how are you and namaste")
 
    while True:
        print("Listening....")
        query = takecommand()
        sites = [ ["youtube", "https://www.youtube.com"], ["google", "https://www.google.com"], ["wikipedia", "www.wikipedia.com"]]
        for site in sites:
            if f"Open {site[0]}".lower() in query.lower():
                speaker.speak(f"opening {site[0]}")
                webbrowser.open(site[1])
            # speaker.speak(query)
        if "open music" in query:
            musicpath = "D:\GLOBAL\audiorecognition\Egzod & Maestro Chives - Royalty (Don Diablo Remix) [NCS Release].mp3"
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, musicpath])
