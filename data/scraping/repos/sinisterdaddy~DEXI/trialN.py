import pyttsx3
import datetime
import speech_recognition as sr
import pyaudio
import smtplib
import geocoder
import wikipedia
#from newsapi import NewsApiClient
import newsapi
import pywhatkit
import requests
# from loginCREDS import senderMAIL, pwd, to
import pyautogui
import webbrowser as wb
import clipboard
import os
import time as tt
import pyjokes
import string
import subprocess as sp
import random
from time import sleep
import openai
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QTimer, QTime, QDate, Qt
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUiType
import sys
from guimain import Ui_MainWindow
# import pygame
import serial
ser = serial.Serial('COM3', 9600)
import pandas as pd
df = pd.read_csv('Active Faculty DataN.csv')


engine = pyttsx3.init()     # Engine property modifications
engine.setProperty('rate', 235)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

cal = 0 # Current Authentication Level

# def speak(text):
# voice = 'en-US-SteffanNeural'
# data = f'edge-tts --voice "{voice}" --text "{text}" --write-media data.mp3'
# os.system(data)

# pygame.init()
# pygame.mixer.init()
# pygame.mixer.music.load("data.mp3")

# try:
#     pygame.mixer.music.play()

#     while pygame.mixer.music.get_busy():
#         pygame.time.Clock().tick(10)

# except Exception as e :
#     print(e)

# finally:
#     pygame.mixer.music.stop()
#     pygame.mixer.quit()


def speak(text):        # Function to Speak
    engine.say(text)
    engine.runAndWait()


def time():     # Function to "Speak" Time
    currTime = datetime.datetime.now().strftime("%I:%M:%S")
    speak("Current Time is: ")
    speak(currTime)


def date():     # Function to "Speak" Date
    months = {1: 'Janauary',
              2: 'February',
              3: 'March',
              4: 'April',
              5: 'May',
              6: 'June',
              7: 'July',
              8: 'August',
              9: 'September',
              10: 'October',
              11: 'November',
              12: 'December'
              }
    dates = {1: 'First',
                2: 'Second',
                3: 'Third',
                4: 'Fourth',
                5: 'Fifth',
                6: 'Sixth',
                7: 'Seventh',
                8: 'Eighth',
                9: 'Ninth',
                10: 'Tenth',
                11: 'Eleventh',
                12: 'Twelfth',
                13: 'Thirteenth',
                14: 'Fourteenth',
                15: 'Fifteenth',
                16: 'Sixteenth',
                17: 'Seventeenth',
                18: 'Eighteenth',
                19: 'Nineteenth',
                20: 'Twentieth',
                21: 'Twenty-First',
                22: 'Twenty-Second',
                23: 'Twenty-Third',
                24: 'Twenty-Fourth',
                25: 'Twenty-Fifth',
                26: 'Twenty-Sixth',
                27: 'Twenty-Seventh',
                28: 'Twenty-Eighth',
                29: 'Twenty-Ninth',
                30: 'Thirtieth',
                31: 'Thirty-First'
             }

    year = str(datetime.datetime.now().year)
    month = int(datetime.datetime.now().month)
    month = months[month]
    date = int(datetime.datetime.now().day)
    date = dates[date]
    text = "Today's Date is "+date+" of "+month+" "+year
    text = str(text)

    speak(text)


def greet():        # Function to "Speak" a greeting pertaining to the current time
    hour = datetime.datetime.now().hour
    if hour <= 6 and hour < 12:
        speak("Good Morning to you!")

    elif hour >= 12 and hour < 18:
        speak("Good Afternoon to you!")

    elif hour >= 18 and hour < 24:
        speak("Good Evening to you!")

    else:
        speak("Good Night to you!")


def wishme():       # Funtion to "Speak" Initial Lines
    speak("HELLO YOU!")
    date()
    time()
    greet()
    speak("Dexi at your service!")


def takeCMD():      # Function to take CMD Commands input
    query = input("Here, input the CMD Command you want: ")
    return query


def takeMIC():      # Function to take MIC inputs
    r = sr.Recognizer()
    listenCMD = [
        'I am listening to you now...',
        'Go on, say something...',
        'Ask away, I am here...',
        'Tell me what do you need?',
        'Whats up?'
    ]
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        speak(random.choice(listenCMD))
        print("Listening to you now...")
        r.pause_threshold = 1
        audio = r.listen(source)
    try:
        print("I heard that ")
        query = r.recognize_google(audio, language="en-IN")
        print(query)
        print("You said: "+query)
    except Exception as e:
        print(e)
        speak("Say that again please...")
        return "Didn't get you, sorry"
    return query


# def sendEmail():        # Function to send E-Mails
#     server  = smtplib.SMTP_SSL('smtp.gmail.com', 465)
#     server.starttls()
#     server.login(senderMAIL, pwd)
#     server.sendmail(senderMAIL, to, 'test mail hai bhay')
#     server.close()

def sendWAPPmsg(pnum, msg):        # Function to send WhatsApp messages
    message = msg
    wb.open('https://web.whatsapp.com/send?phone='+pnum+'&text='+message)
    sleep(10)
    pyautogui.press('enter')


def searchGGL(res):        # Function to search a topic on Google
    wb.open('https://www.google.com/search?q='+res)


def weather():      # Function to look up current Weather Conditions
    g = geocoder.ip('me')
    lat = str(g.lat)
    lon = str(g.lng)
    wkey = '9bbd336b60745f2810e848836ee7fc11'
    lnk = 'https://api.openweathermap.org/data/2.5/weather?lat=' + \
        lat+'&lon='+lon+'&appid='+wkey
    print(lnk)
    return lnk


# def news(topic):        # Function to get top news headlines on specific topic
#     newsapi = NewsApiClient(api_key='424b4d1da9614c749f2b783188acf6a9')
#     data = newsapi.get_top_headlines(q=topic,
#                                      language='en',
#                                      page_size=5)
#     ndata = data['articles']
#     for x, y in enumerate(ndata):
#         print(f'{y["description"]}')
#         speak(f'{y["description"]}')

#     speak("that's all for today.")


def t2s():        # Function to read clipboard data
    text = clipboard.paste()
    print(text)
    speak(text)


def open_cmd():
    os.system('start cmd')


def open_notepad():
    notepad = "C:\\Windows\\System32\\notepad.exe"
    os.startfile(notepad)


def open_camera():
    sp.run('start microsoft.windows.camera:', shell=True)


def open_calculator():
    calculator = "C:\Windows\System32\calc.exe"
    sp.Popen(calculator)


def get_random_advice():
    res = requests.get("https://api.adviceslip.com/advice").json()
    res = res['slip']['advice']
    print(res)
    speak(res)

# def snap():        # Function to take a screenshot
#     name_img = tt.time()
#     name_img = f'C:\\work (v boring)\\ecs\\screenshot\\{name_img}.png'
#     img = pyautogui.screenshot(name_img)
#     speak("Took the screenshot!")


def pwdGEN():       # Function to Generate a new password
    s1 = string.ascii_uppercase
    s2 = string.ascii_lowercase
    s3 = string.digits
    s4 = string.punctuation
    pwdLEN = 8

    s = []
    s.extend(list(s1))
    s.extend(list(s2))
    s.extend(list(s3))
    s.extend(list(s4))

    random.shuffle(s)
    newPASS = ("".join(s[0:pwdLEN]))

    print(newPASS)
    speak(newPASS)


def gpt(q):
    openai.api_key = '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'

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": q}
        ]
    )

    cnt = completion.choices[0].message
    val = cnt["content"]
    val = val.replace('"', '')

    print(val)
    speak(val)


def dc(q):
    for i, r in df.iterrows():
        if q.isnumeric():
            if q == str(r['Emp. ID']):
                n = r['Name of the Faculty']
                des = r['Designation']
                dpt = r['Department']
                sch = r['School']
                mob = r['Mobile No']
                eml = r['E-mail']
                cab = r['Cabin']
                blk = r['Block']
                s = "Name: "+n+", Designation: " + des + ", Department: " + dpt + ", School: " + sch + \
                    ", Mobile No: " + mob + ", E-mail: " + eml + \
                    ", Block: " + blk + ", Cabin: " + cab
                print(str(s))
                sp = "The cabin number of " + n + " is at the " + blk + \
                    " " + cab + " and you can contact them at: " + eml
                speak(sp)

        else:
            if q == str(r['Name of the Faculty']):
                eid = r['Emp. ID']
                des = r['Designation']
                dpt = r['Department']
                sch = r['School']
                mob = r['Mobile No']
                eml = r['E-mail']
                cab = r['Cabin']
                blk = r['Block']
                s = "ID: "+str(eid)+", Designation: " + des + ", Department: " + dpt + ", School: " + \
                    sch + ", Mobile No: " + mob + ", E-mail: " + \
                    eml + ", Block: " + blk + ", Cabin: " + cab
                print(str(s))
                sp = "The cabin number of " + q + " is at the " + blk + \
                    " " + cab + " and you can contact them at: " + eml
                speak(sp)


def faculty(str):
    dc(str)


def guide(str):
    dest = {
        'health centre': 'A B - 1, Ground Floor',
        'Einstien hall': 'A B - 1, First Floor',
        'library': 'A B - 1, Second Floor',
        'Newton hall': 'A B - 1, Fourth Floor',
        'amphitheatre': 'A B - 2, Back Side',
        'auditorium': 'A B - 2, Third Floor',
        'vit school of law': 'A B - 2, Fourth Floor',
        'incubation centre': 'C B - Second Floor',
        'vice chancellor office': 'C B - First Floor',
        'centre of excellence': 'C B - Second Floor'
    }
    res = "The " + str + " is located at the " + dest[str]
    print(res)
    speak(res)

def authCheck(u, p):
    df = pd.read_csv("empAuth.csv")
    for i in df.itertuples():
        if i[1]==u and i[3]==p:
            ol=i[4]
            return ol
        else:
            ol=0
            return ol
    


def startup():
    speak("Initializing Dexi")
    speak("Starting all systems applications")
    # speak("Installing and checking all drivers")
    # speak("Caliberating and examining all the core processors")
    # speak("Checking the internet connection")
    # speak("Wait a moment sir")
    # speak("All drivers are up and running")
    # speak("All systems have been activated")
    speak("Now I am online")
    hour = int(datetime.datetime.now().hour)
    if hour >= 0 and hour <= 12:
        speak("Good Morning")
    elif hour > 12 and hour < 18:
        speak("Good afternoon")
    else:
        speak("Good evening")
    
    speak("Please enter your Employee ID and Password: ")
    usr = int(input("Enter Employee ID: "))
    pwd = input("Enter Password: ")
    authLVL = authCheck(usr, pwd)
    cal = authLVL
    if authLVL==0:
        speak("Sorry, I do not recognise you, are you sure you put in the correct credentials?")

    speak("I am Dexi. Online and ready sir. Please tell me how may I help you")

def wkp():
    speak("Hey there!")
    speak("Dexi is here to assist you with all your needs! Ask away")

#if __name__ == "__main__":        # Main Function

    # wishme()
    # greet()
class MainThread(QThread):
    def __init__(self):
        super(MainThread, self).__init__()

    def run(self):
        self.TaskExecution()
        
    def TaskExecution(self):
        startup()

        while True:
            query = takeMIC().lower()

            if 'wake up' in query:
                wkp()
                
            if 'time' in query:
                time()

            elif 'date' in query:
                date()

            elif 'message' in query:
                if cal == 1:
                    user_name = {
                        'Dexi': '+91 93580 84318'
                        }
                    try:
                        speak("To whom do you want me to text for you?")
                        name = takeMIC()
                        pno = user_name[name]
                        speak("What should I text?")
                        msg = takeMIC()
                        sendWAPPmsg(pno, msg)
                        speak("The message is sent!")
                    except Exception as e:
                        print(e)
                        speak("Sorry, I was unable to send the text.")
                else:
                    speak("Sorry, you do not have clearance for this action!")
                    print("Sorry, you do not have clearance for this action!")

            elif 'wikipedia' in query:
                speak("Working on it.")
                query = query.replace("Wikipedia", "")
                res = wikipedia.summary(query, sentences=2)
                print(res)
                speak(res)

            elif 'Google' in query:
                res = query.replace("Google", "")
                searchGGL(res)

            elif 'youtube' in query:
                speak("What do you wanna look up on YouTube?")
                q = takeMIC()
                pywhatkit.playonyt(q)

            elif 'weather' in query:
                url = weather()
                res = requests.get(url)
                data = res.json()
                wtr = data['weather'][0]['main']
                tmp = data['main']['temp']
                tmp = int(tmp) - 273
                dis = data['weather'][0]['description']
                speak("The weather conditions right now are "+wtr)
                speak("The current temperature is "+str(tmp)+" degrees Celsius.")
                speak(dis)
                print(wtr)
                print(tmp)
                print(dis)

            elif 'guide me' in query:
                speak('Where do you want me to guide you to?')
                des = takeMIC()
                guide(des)

            elif 'faculty' in query:
                speak("Enter the name of the faculty you need the details off: ")
                inp = input()
                faculty(inp)
            elif 'tell me' in query:
                gpt(query)
            
            # elif 'news' in query:
            #     speak("What topic do you want the news for?")
            #     res = takeMIC()
            #     news(res)

            elif 'read' in query:
                t2s()

            elif 'open notepad' in query:
                if cal == 1:
                    open_notepad()
                else:
                    speak("Sorry, you do not have clearance for this action!")
                    print("Sorry, you do not have clearance for this action!")

            elif 'open cmd' in query:
                if cal == 1:
                    open_cmd()
                else:
                    speak("Sorry, you do not have clearance for this action!")
                    print("Sorry, you do not have clearance for this action!")

            # elif 'open code' in query:
            #     pth = 'C:\Users\krish\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Visual Studio Code'
            #     os.startfile(pth)

            elif 'open files' in query:
                if cal == 1:
                    os.system('explorer C://')
                else:
                    speak("Sorry, you do not have clearance for this action!")
                    print("Sorry, you do not have clearance for this action!")

            elif 'open camera' in query:
                if cal == 1:
                    open_camera()
                else:
                    speak("Sorry, you do not have clearance for this action!")
                    print("Sorry, you do not have clearance for this action!")

            elif 'open calculator' in query:
                if cal == 1:
                    open_calculator()
                else:
                    speak("Sorry, you do not have clearance for this action!")
                    print("Sorry, you do not have clearance for this action!")

            elif 'joke' in query:
                speak(pyjokes.get_joke())

            elif 'advice' in query:
                get_random_advice()

            # elif 'snapshot' or 'screenshot' in query:
            #     snap()

            elif 'remember that' in query:
                str = query.replace('remember that', '')
                remember = open('data.txt', 'w')
                remember.write(str)
                remember.close()
                speak("Alright, I shall remember that "+str)

            elif 'remind me about' in query:
                remember = open('data.txt', 'r')
                speak("you told me to remember that"+remember.read())
                speak("That was it.")

            elif 'generate a password' in query:
                pwdGEN()

            elif 'tell me' in query:
                gpt(query)

            elif 'all on' in query:
                if cal == 1:
                    speak('okay sir turning ON')
                    ser.write(b'A')
                else:
                    speak("Sorry, you do not have clearance for this action!")
                    print("Sorry, you do not have clearance for this action!")

            elif 'all of' in query:
                if cal == 1:
                    speak('okay sir turning Off')
                    ser.write(b'a')
                else:
                    speak("Sorry, you do not have clearance for this action!")
                    print("Sorry, you do not have clearance for this action!")
                
            elif 'bulb on' in query:
                if cal == 1:
                    speak('okay sir the bulb turning ON')
                    ser.write(b'W')
                else:
                    speak("Sorry, you do not have clearance for this action!")
                    print("Sorry, you do not have clearance for this action!")

            elif 'bulb of' in query:
                if cal == 1:
                    speak('okay sir turning the bulb Off')
                    ser.write(b'Q')
                else:
                    speak("Sorry, you do not have clearance for this action!")
                    print("Sorry, you do not have clearance for this action!")
                
            elif 'light on' in query:
                if cal == 1:
                    speak('okay sir turning the light ON')
                    ser.write(b'R')
                else:
                    speak("Sorry, you do not have clearance for this action!")
                    print("Sorry, you do not have clearance for this action!")

            elif 'light of' in query:
                if cal == 1:
                    speak('okay sir turning the light Off')
                    ser.write(b'E')
                else:
                    speak("Sorry, you do not have clearance for this action!")
                    print("Sorry, you do not have clearance for this action!") 
                
            elif 'fan on' in query:
                if cal == 1:
                    speak('okay sir turning the fan ON')
                    ser.write(b'Y')
                else:
                    speak("Sorry, you do not have clearance for this action!")
                    print("Sorry, you do not have clearance for this action!")

            elif 'fan of' in query:
                if cal == 1:
                    speak('okay sir turning the fan Off')
                    ser.write(b'T')
                else:
                    speak("Sorry, you do not have clearance for this action!")
                    print("Sorry, you do not have clearance for this action!") 

            elif 'offline' or 'sleep' or 'bye'  or 'exit' in query:
                speak("Alright! Have a good one, buh byee!")
            exit()    
            
startExecution = MainThread()


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.startTask)
        self.ui.pushButton_2.clicked.connect(self.close)

    def __del__(self):
        sys.stdout = sys.__stdout__

    def run(self):
        self.TaskExection
    def startTask(self):
        self.ui.movie = QtGui.QMovie("C:/Users/krish/Documents/RK/PROJECTS_RK/DEXI/DEXI/images/gif.gif")
        self.ui.label.setMovie(self.ui.movie)
        self.ui.movie.start()

        timer = QTimer(self)
        timer.timeout.connect(self.showTime)
        timer.start(1000)
        startExecution.start()

    def showTime(self):
        current_time = QTime.currentTime()
        current_date = QDate.currentDate()
        label_time = current_time.toString('hh:mm:ss')
        label_date = current_date.toString(Qt.ISODate)
        self.ui.textBrowser.setText(label_date)
        self.ui.textBrowser_2.setText(label_time)


app = QApplication(sys.argv)
Dexi = Main()
Dexi.show()
exit(app.exec_())
