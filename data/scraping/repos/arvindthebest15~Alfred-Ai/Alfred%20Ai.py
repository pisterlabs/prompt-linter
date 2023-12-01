import speech_recognition as sr
import os
import webbrowser
import openai
import datetime

def say(text):
    os.system(f"say {text}")
def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        #r.pause_threshold = 0.6
        audio = r.listen   (source)
        try:
            print("Recognizing...")
            query = r.recognize_google(audio, language="en-in")
            print(f"User said: {query}")
            return query
        except Exception as e:
            return "Some Error Occurred. Sorry From Alfred"

if __name__ == '__main__':
    print('PyCharm')
    say("Hello I am Alfred AI")
    while True:
        print("Listening...")
        query = takeCommand()
        # todo: add more sites
        sites = [["Youtube", "https://www.youtube.com"], ["Wikipedia", "https://www.wikipedia.com"], ["Google", "https://www.google.com"],["Twitter", "https://www.twitter.com"], ["My Portfolio", "https://arvindnadar.netlify.app/#"],["jw.org", "https://www.jw.org/en/"]]
        for site in sites:
            if f"Open {site[0]}".lower() in query.lower():
                say(f"Master Arvind, Alfred is Opening {site[0]} for you")
                webbrowser.open(site[1])

            #todo: add a feature to play a specific song

            if "open music" in query:
                musicPath = "/Users/mac/Downloads/Eminem-Till-I-Collapse-via-Naijafinix.com_.mp3"
                os.system(f"open {musicPath}")

            if "the time" in query:
                hour = datetime.datetime.now().strftime("%H")
                minute = datetime.datetime.now().strftime("%M")
                say(f"Master Arvind the time is {hour} hour {minute} minute")
                break

            if "date" in query:
                day = datetime.datetime.now().strftime("%A")
                month = datetime.datetime.now().strftime("%B")
                date = datetime.datetime.now().strftime("%d")
                year = datetime.datetime.now().strftime("%Y")
                say(f"Master Arvind today is {day} {month} {date} {year}")
                break

            if "open facetime".lower() in query.lower():
                os.system(f"open /System/Applications/FaceTime.app")

            if "open calender".lower() in query.lower():
                os.system(f"open /System/Applications/Calendar.app")



        #say(query)