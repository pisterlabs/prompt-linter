#MAC BASED
#Download pyaudio too
import os
import webbrowser
import speech_recognition as sr
import wikipedia
#import openai
import datetime


def say(text):
    os.system(f"say {text}")

def takeCommand():
    r=sr.Recognizer()
    with sr.Microphone() as source:
        r.pause_threshold=1
        audio=r.listen(source)
        try:
            print("Recognizing....")
            query = r.recognize_google(audio, language="eng-in")
            print(f"User said: {query}")
            return query
        except Exception as e:
            return "Some Error Occurred. Sorry from Jarvis"


if __name__== '__main__':
    say("hello I am jarvis A.I. ")
    while True:
     print("Listening.....")
     query = takeCommand()
     say(query)
     sites=["youtube","https://www.youtube.com"],["wikipedia","https://www.wikipedia.com"],["google","https://www.google.com"]
     for site in sites:
         if f"Open {site[0]}".lower() in query.lower():
             say(f"Opening  {site[0]} sir...")
             webbrowser.open(site[1])
     #if "open music" in query:
         #musicPath=""
         #os.system(f"open{musicPath}")
     if "the time" in query:
         strfTime=datetime.datetime.now().strftime("%H:%H:%S")
         say(f"Sir the time is {strfTime}")

     if "open facetime".lower() in query.lower():
         os.system(f"open /System/Application/FaceTime.app")

     if "conscious" or "self aware" in query:
         say("That's a difficult question. Can you prove that you are?” And I bet you cannot do so")
