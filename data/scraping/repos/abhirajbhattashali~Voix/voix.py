import speech_recognition as sr
import datetime
import wikipedia
import webbrowser
import os
import smtplib
import sys
import re
import pyttsx3
import wolframalpha
from googlesearch import search
import openai
import CONSTANTS
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voices', voices[0].id)


class Voix:


    def __init__(self):
        self.speak("Initiating Voix...")
        self.speak("All systems running...")
        self.wishMe()

    def speak(self, text):
        try:
            engine.say(text)
            engine.runAndWait()
            engine.setProperty('rate', 175)
            return True
        except:
            t = "Sorry I couldn't understand and handle this input"
            print(t)
            return False

    def computational_intelligence(self, question):
        try:
            client = wolframalpha.Client(CONSTANTS.AppId)
            answer = client.query(question)
            answer = next(answer.results).text
            print(answer)
            return answer
        except:
            self.speak("Sorry I couldn't fetch your question's answer. Please try again ")
            return None

    def wishMe(self):
        hour = int(datetime.datetime.now().hour)
        if hour >= 0 and hour < 12:
            self.speak("Good Morning!")

        elif hour >= 12 and hour < 18:
            self.speak("Good Afternoon!")

        else:
            self.speak("Good Evening!")

        self.speak("Hi  What can Voix help you with?")

    def current_time(self):
        c_time = datetime.datetime.now()
        hour = c_time.hour
        min = c_time.minute
        am_or_pm = c_time.strftime("%p")
        time = f"It's {hour}:{min} {am_or_pm} now"
        print(time)
        self.speak(time)

    def current_date(self):
        date = datetime.datetime.now()
        day = date.strftime("%A")
        dayNo = date.strftime("%d")
        month = date.strftime("%B")
        year = date.strftime("%Y")
        date_str = f"Today is {day} {dayNo} {month} {year}"
        print(date_str)
        self.speak(date_str)

    def sendEmail(self, to, msg):
        if __name__ == '__main__':
            my_mail = CONSTANTS.Mail
            password = CONSTANTS.Password
            port = 587
            smtp_server = 'smtp.gmail.com'

        try:
            server = smtplib.SMTP(smtp_server, port)
            server.starttls()
            server.login(user=my_mail, password=password)
            server.sendmail(
                from_addr=my_mail,
                to_addrs=to,
                msg=msg
            )
            print("Successfully sent mail")
            self.speak("Successfully sent mail")

            server.quit()
        except Exception as e:
            print(e)

    def takeCommand(self):
        r = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            print("Listening...")
            self.speak("Listening")

            r.pause_threshold = 0.7
            audio = r.listen(source)

        try:
            print("Recognizing...")
            self.speak("Recognizing")
            query = r.recognize_google(audio, language='en-in')
            print(f"User said: {query}\n")
        except Exception:
            self.speak("Sorry I didn't get that could u repeat")
            print("Sorry I didn't get that could u repeat...")
            return "None"
        return query

    def website_opener(self, domain):
        try:
            url = 'https://www.' + domain
            webbrowser.open(url)
            return True
        except Exception as e:
            print(e)
            return False

    def google_search(self, query):
        for res in search(query, tld='com', num=1, stop=1, pause=2):
            print(res)

        if res:
            webbrowser.open(res)
        else:
            self.speak("Nothing found on web!")
            print("Nothing found on web!")


def gpt3(text):
    openai.api_key = CONSTANTS.Api_key
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=text,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )


    content = response.choices[0].text
    return content

voix = Voix()

while True:
    q = input("Enter query: ")
    query = voix.takeCommand().lower()

    if re.search("today's date", query) or re.search("today", query) or re.search("date", query):
        voix.current_date()

    elif re.search("open google", query):
        voix.speak("Opening Google")
        webbrowser.open("https://www.google.com/")

    elif re.search("open youtube", query):
        voix.speak("Opening Youtube")
        webbrowser.open("https://www.youtube.com/")

    elif re.search("wikipedia", query):
        query = re.sub("wikipedia", "", query)
        print(query)
        results = wikipedia.summary(query, sentences=2)
        voix.speak("According to Wikipedia")
        print(results)
        voix.speak(results)

    elif re.search('send mail', query):
        voix.speak("Enter the name of receipient:")
        receipient = input("Recepeint id: ")
        ans = "yes"
        while ans in ["yes", "Yes", "YES"]:
            voix.speak("What's the message you'd like to send.. ")
            msg = input("Enter message: ")
            voix.speak(f"Do u wish to make any changes with the message {msg}")
            ans = input("Want to make any changes: (Yes or No):")
        voix.sendEmail(receipient, msg)

    elif re.search("time", query):
        voix.current_time()

    elif re.search("open", query):
        domain = query.split()[-1]
        voix.website_opener(domain)

    elif re.search("exit", query) or re.search("quit", query):
        voix.speak("Exiting Voix")
        sys.exit()


    elif re.search("calculate", query):
        question = query
        answer = voix.computational_intelligence(question)
        voix.speak(answer)

    elif re.search("what is", query) or re.search("who is", query):
        question = query
        answer = voix.computational_intelligence(question)
        voix.speak(answer)

    elif re.search("search", query):
        query = re.sub("search", "", query)
        voix.google_search(query)

    elif re.search("search gpt",query):
        response = response.relace("search gpt","")
        response = gpt3(query)
        print(response)