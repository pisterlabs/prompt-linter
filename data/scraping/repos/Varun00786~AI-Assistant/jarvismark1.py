from googletrans import Translator
import pyjokes
import pyttsx3
import subprocess
import speech_recognition as sr
import wikipedia
import datetime
import webbrowser
import speedtest
import os,sys
import pywhatkit
import requests
from bs4 import BeautifulSoup
from PyQt5 import QtGui
# from PyQt5.QtGui import QMovie
from PyQt5.QtCore import QTimer,QTime,QDate,Qt
from PyQt5.QtCore import *
from PyQt5.QtGui import *
# from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import *
# from PyQt5.uic import loadUiType
from jarvisui import Ui_Jarvis
from jarvisui import *
import pyautogui
import alarm
import openai
from dotenv import load_dotenv

fileopen=open("data.txt","r")
Api=fileopen.read()
fileopen.close()

openai.api_key=Api
load_dotenv()
comp=openai.Completion()
# import time
import gradio as gr
openai.api_key =Api

engine = pyttsx3.init()
voices= engine.getProperty('voices') #getting details of current voice
engine.setProperty('voice', voices[0].id)
def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def Wishme():
    # Time=datetime.datetime.now().strftime("%H:%M")
    # Time=Time.replace(":","hours")
    # # Time=Time.replace("0","")
    # Time=Time+"minutes"
    Time1=datetime.datetime.now().hour
    Time2=datetime.datetime.now().minute
    speak(f"hello iam jarvis its,{Time1} hours and {Time2} minutes how can i help you")            

def trans(Text):
    line=str(Text)
    translate=Translator()
    result=translate.translate(Text)
    query=result.text
    print(query)
    return query

def rplybrain(ques,chat_log=None):
    FileLog=open("reply.txt","r")
    lognew=FileLog.read()
    FileLog.close()

    if chat_log is None:
        chat_log=lognew
    
    prompt=f'{chat_log}You: {ques}\nJarvis :'
    response= comp.create(
        model ="text-davinci-002",
        prompt=prompt,
        temperature=0.5,
        max_tokens=60,
        top_p=0.3,
        frequency_penalty=0.5,
        # presense_penalty=0
    )
    answer= response.choices[0].text.strip()
    logupdate=lognew+f'\nYou : {ques} \nJarvis : {answer}'
    FileLog=open("reply.txt",'w')
    FileLog.write(logupdate)
    FileLog.close()
    return answer


def openai_chat(prompt):
    completions = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        temperature=0.5,
    )

    message = completions.choices[0].text
    return message.strip()

def chatbot(input, history=''):
    output = openai_chat(input)
    history=output
    return history

def loc():
    speak("searching location")
    ip_add=requests.get("http://api.ipify.org").text
    url="https://get.geojs.io/v1/ip/geo/"+ip_add+".json"
    geo_q=requests.get(url)
    geo_q=geo_q.json()
    state=geo_q["city"]
    country=geo_q["country"]
    speak(f"you are in {state,country}")  
class MainThread(QThread):
    def __init__(self):
        super(MainThread,self).__init__()
        
        
    def run(self):
        self.run1()
        
    
    def takeCommand(self):
        #It takes microphone input from the user and returns string output
        self.query=""
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            speak("Listening")
            # r.pause_threshold = 1
            r.energy_threshold = 500
            # r.adjust_for_ambient_noise(source,10)
            audio = r.listen(source)

        try:
            print("recognizing....")
            speak("recognizing.")
            self.query=r.recognize_google(audio,language="en-in")
            self.query=trans(self.query)
            print(f"User said: {self.query}\n")
            speak("did you say,"+self.query)
        except Exception as e:
            print(e)
            speak("say it again please")
            print("say it again please...")
            
            return "None"
        return self.query
    def run1(self):
        Wishme()
        while True:
            self.query=self.takeCommand().lower()
            if "wikipedia" in self.query:
                speak("Searching wikipedia....")
                self.query=self.query.replace("wikipedia","")
                results=wikipedia.summary(self.query,sentences=2)
                speak("According to wikipedia..")
                print(results)
                speak(results)
            elif "open youtube" in self.query:
                speak("Opening youtube")
                webbrowser.open("youtube.com")  
            elif "open epic games" in self.query:
                speak("Opening epic games")
                webbrowser.open("https://store.epicgames.com/en-US/.com")  
            elif "what is the time" in self.query:
                strTime=datetime.datetime.now().strftime("%H:%M:%S")
                speak(f"The time is {strTime}")
            elif "open vs code" in self.query: 
                vspath="C:\\Microsoft VS Code\\Code.exe" 
                os.startfile(vspath)
            elif "open chrome" in self.query: 
                vspath="C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" 
                speak("Opening chrome")
                os.startfile(vspath)
            elif "open epic launcher" in self.query: 
                vspath="C:\\Program Files (x86)\\Epic Games\\Launcher\\Portal\\Binaries\\Win32\\EpicGamesLauncher.exe"
                speak("Opening epic launcher")
                os.startfile(vspath)
            elif "open whatsapp" in self.query: 
                vspath="C:\\Users\\ASUS\\AppData\\Local\\WhatsApp\\WhatsApp.exe"
                speak("Opening whatsapp")
                os.startfile(vspath)
            elif "open idea" in self.query: 
                vspath="C:\\Program Files\\JetBrains\\IntelliJ IDEA Community Edition 2022.1\\bin\\idea64.exe"
                speak("Opening Intellij")
                os.startfile(vspath)
            elif "open pycharm" in self.query: 
                vspath="C:\\Program Files\\JetBrains\\PyCharm Community Edition 2022.1.2\\bin\\pycharm64.exe"
                speak("Opening Pycharm")
                os.startfile(vspath)
            elif "search google" in self.query:
                self.query= self.query.replace("search google", "")
                search = self.query
                speak("searching"+search+"on google")
                webbrowser.open("https://www.google.com/search?q=" + search + "")
            elif 'joke' in self.query:
                joke = pyjokes.get_joke()
                speak(joke)
            elif "open camera" in self.query or "take a photo" in self.query:
                speak("Opening Camera")
                subprocess.run('start microsoft.windows.camera:', shell=True)
            
            elif 'volume up' in self.query:
                speak("increasing volume")
                # time.sleep(1)
                for i in range(0,5):
                    pyautogui.press("volumeup")
            elif 'volume down' in self.query:
                speak("decreasing volume")
                for i in range(0,5):
                    pyautogui.press("volumedown")   
            elif "open document" in self.query:
                speak("Opening documents")
                subprocess.Popen('explorer "C:\\temp"')
            elif "where is" in self.query:
                self.query = self.query.replace("where is", "")
                location = self.query
                speak("Tracing Location")
                speak(location)
                url = "https://www.google.nl/maps/place/" + location + ""
                webbrowser.open(url)
            elif "open notepad" in self.query:
                speak("Opening Notepad")
                os.startfile('C:\\Windows\\notepad.exe')
            elif "open explorer" in self.query:
                speak("Opening File Explorer")
                os.startfile('C:\\Windows\\explorer.exe')
            elif "open excel" in self.query:
                speak("Opening Excel")
                os.startfile('C:\\Program Files\\Microsoft Office\\Office16\\EXCEL.exe')
            elif "open powerpoint" in self.query:
                speak("Opening Power point")
                os.startfile('C:\\Program Files\\Microsoft Office\\Office16\\POWERPNT.exe')
            elif "open word" in self.query:
                speak("Opening wordpad")
                os.startfile('C:\\Program Files\\Microsoft Office\\Office16\\WINWORD.exe')
            elif "open edge" in self.query:
                speak("Opening Edge")
                os.startfile('C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe')
            
            elif "check speed" in self.query:
                speak("checking speed")
                self.speed=speedtest.Speedtest()
                self.down=self.speed.download()
                self.downspd=int(int(self.down)/80000)
                speak(f"your downloding speed is {self.downspd}Mb/s")
            elif 'play' in self.query:
                song = self.query.replace('play', '')
                speak('playing' + song)
                pywhatkit.playonyt(song)
            elif 'say' in self.query:
                say = self.query.replace('say', '')
                speak(say)
            elif 'search location' in self.query or "my location" in self.query:
                loc()
            
            elif "open music" in self.query or "open spotify" in self.query:
                speak("Opening Spotify")
                os.startfile("C:\\Users\\ASUS\\AppData\\Roaming\\Spotify\\Spotify.exe")
            elif "open command" in self.query:
                speak("opening cmd")
                os.system("start cmd")
            elif "temperature" in self.query or "weather" in self.query:
                search=self.query      #.replace("temperature","")
                url=f"https://www.google.com/search?q={search}"
                r=requests.get(url)
                data=BeautifulSoup(r.text,"html.parser")
                temp=data.find("div",class_="BNeawe").text
                speak(f"current{search} is {temp}")
            elif "close "in self.query:
                self.query=self.query.replace("close","")
                speak(f"okay i'm,closing {self.query}")
                os.system(f"taskkill /f /im {self.query}.exe")
            elif"close edge" in self.query:
                speak(f"okay i'm,closing edge")
                os.system(f"taskkill /f /im msedge.exe")
            elif "set alarm"in self.query:
                speak("sir please tell me the time to set alarm. for example set alarm to 5:45 pm")
                self.tt=self.takeCommand().upper()
                self.tt=self.tt.replace("SET ALARM TO ","")
                self.tt=self.tt.replace(".","")
                print(self.tt)
                alarm.alarm(self.tt)
            elif "check network " in self.query:
                speak("checking speed")
                self.speed=speedtest.Speedtest()
                self.down=self.speed.download()
                self.downspd=int(int(self.down)/80000)
                speak(f"your downloding speed is {self.downspd}")
            elif "advanced mode " in self.query:
                speak("ask anything")
                dk=self.takeCommand().lower()
                speak(chatbot(dk))
            elif "exit" in self.query:
                speak("Shutting Down")
                self.close();
          
            else:
                kk=rplybrain(self.query)
                speak(kk)

startExecution= MainThread()
# c=startExecution.takeCommand()
# d=startExecution.run1() 

class Main(QMainWindow,Ui_Jarvis):
    

    def __init__(self,parent=None):
        super(Main,self).__init__(parent)
        self.setupUi(self)
        self.movie=QtGui.QMovie("../../Users/ASUS/Pictures/ironman/jarvis.gif")
        self.label.setMovie(self.movie)
        self.movie.start()
        self.movie=QtGui.QMovie("../../Users/ASUS/Pictures/ironman/Jarvis_Loading_Screen.gif")
        self.label_2.setMovie(self.movie)
        self.movie.start()
        self.pushButton.clicked.connect(self.startTask)
        self.pushButton_2.clicked.connect(self.close)
        

    
    def showtime(self):
        current_time=QTime.currentTime()
        current_date=QDate.currentDate()
        label_time=current_time.toString("hh:mm:ss")
        label_date=current_date.toString(Qt.ISODate)
        self.textBrowser.setText(label_date)
        self.textBrowser_2.setText(label_time)

    def startTask(self):
        
        timer=QTimer(self)
        timer.timeout.connect(self.showtime)
        timer.start(1000)
        startExecution.start(1000)
        
        


app=QApplication(sys.argv)
jarvisui=Main()
jarvisui.show()
exit(app.exec_())