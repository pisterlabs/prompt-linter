from pkgutil import ImpImporter
import random
import pygame
import speech_recognition as sr
import sys 
import pyjokes
import datetime
from datetime import datetime

import os
import time
import subprocess
import wolframalpha

import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog

import json
import operator
import wikipedia
import webbrowser
import winshell
import feedparser
import smtplib
import ctypes
import requests
import shutil
import pyaudio
from twilio.rest import Client
from clint.textui import progress
from ecapture import ecapture as ec
from bs4 import BeautifulSoup
import win32com.client as wincl
from urllib.request import urlopen
from pygame import mixer
import docx

import os

import openai
import pygame
import requests

import shutil
from craiyon import Craiyon
from PIL import Image # pip install pillow
from io import BytesIO
import base64

import io
import warnings
import docx

import  pywhatkit
import glob




recognizer = sr.Recognizer()







def moveAllFilesinDir(srcDir, dstDir):
    # Check if both the are directories
    if os.path.isdir(srcDir) and os.path.isdir(dstDir) :
        # Iterate over all the files in source directory
        for filePath in glob.glob(srcDir + '\*'):
            # Move each file to destination Directory
            shutil.move(filePath, dstDir);
    else:
        print("srcDir & dstDir should be Directories")







def takeCommand():

    
    r = sr.Recognizer()
     
    with sr.Microphone() as source:

            
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)
    
        try:
            print("Recognizing...")   
            query = r.recognize_google(audio, language ='en-us').lower()
            print(query)
    
        except Exception as e:
            print(e)   
            print("Unable to Recognize your voice.") 
            return takeCommand()
        return query



#-------------speakCMD----------------------------------------------

voice2 = 'en-GB-SoniaNeural'
def speak(data):
    voice = 'en-US-SteffanNeural'
    command = f'edge-tts --voice "{voice}" --text "{data}" --write-media "data.mp3"'
    os.system(command)

    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load("data.mp3")

    try:
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    except Exception as e:
        print(e)
    finally:
        pygame.mixer.music.stop()
        pygame.mixer.quit()

#------------END_SPEAKCMD-------------------------------------------

#-----------------WEATHER-------------------------------------------

def weather():
    API_key = "bf462035b98b1d6ef3fcad2b3349f566"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    speak('tell me the name of the city')
    city_name = takeCommand()
    
    Final_url = base_url + "appid=" + API_key + "&q=" + city_name
    weather_data = requests.get(Final_url).json()
    

    Ftemp = weather_data['main']['temp']
    temp = Ftemp-273.15
    
    # Accessing wind speed, it resides in wind and its key is speed
    wind_speed = weather_data['wind']['speed']
    
    # Accessing Description, it resides in weather and its key is description 
    description = weather_data['weather'][0]['main']
    

    
    print(f'\nTemperature is : {temp:.0f}°C')
    print(f'\nWind Speed : {wind_speed}m/s')
    print(f'\nthe sky is : ',description)
    speak(f'Temperature is : {temp:.0f}°C , Wind Speed : {wind_speed},the sky is {description}.')


#-----------------END_WEATHER---------------------------------------

#----------------BRAIN_ELLIOT---------------------------------------

assname="Elliot"
username_file = open("username.json")
uname = json.load(username_file)



            
        
        



def intro():

    choicefile=open("texts/intro.txt","r")
    linelist=[]
    for line in choicefile:
        linelist.append(line)
    choice=random.choice(linelist)

    speak(choice)

def usname_set():
    usname = takeCommand()
    with open("username.json", "w") as f:
        json.dump(""+usname,f)

    



def username():
    time.sleep(0.7)
    username_file = open("username.json")
    uname = json.load(username_file)
    speak("nice to meet you  "+uname+",how can i help you ")






#---------------------



    
def creator():
    choicefile=open("texts/creator.txt","r")
    linelist=[]
    for line in choicefile:
        linelist.append(line)
    choice=random.choice(linelist)
    speak(choice)

def wishMe():
    choicefile=open("texts/hellos.txt","r")
    linelist=[]
    for line in choicefile:
        linelist.append(line)
    choice=random.choice(linelist)
    from datetime import datetime

# datetime object containing current date and time
    now = datetime.now()
    talk =  choice +" How can i Help you this time?"
    hour = int(now.strftime("%H"))
    txt =[]
    txt.append(talk.replace('\n', ' '))
    txtG=str(txt)
    if hour>= 0 and hour<12:
        speak("Good Morning "+uname+' ,'+txtG)  
    
  
    elif hour>= 12 and hour<18:
        speak("Good Afternoon "+uname+' '+txtG) 
        
  
    else:
        speak("Good Evening "+uname+' '+txtG) 

    print(talk)

 
    
def joke():
    funny = pyjokes.get_joke()
    speak(funny)
    print(funny)

def playmusic():
    speak("Here you go with music")
    music_dir = "C:\\Users\\PC\\Music\\Nouveaudossier"
    random0 = os.path.join(music_dir,random.choice(os.listdir(music_dir)))

    mixer.init()
    mixer.music.load(random0)
    mixer.music.play()
    
   
def rndmp3 ():
        randomfile = random.choice(os.listdir("C:\\Users\\PC\\Music\\Nouveaudossier"))
        file = (' C:\\Users\\PC\\Music\\Nouveaudossier'+ randomfile)
        pygame.mixer.init()
        pygame.mixer.music.load(file)
        pygame.mixer.music.play()
def thetime():
    t = time.localtime()
    current_time = time.strftime("%H:%M", t)
    print("now its the "+ current_time)
    speak("now its the "+ current_time)

def brave():
    codePath = r"C:\\Program Files (x86)\\BraveSoftware\\Brave-Browser\\Application\\brave.exe"
    os.startfile(codePath)

def howme():
    choicefile=open("texts/howarme.txt","r")
    linelist=[]
    for line in choicefile:
        linelist.append(line)
    hoarewme=random.choice(linelist)
    speak(hoarewme)
    time.sleep(1)
    speak("what about you ?")
    takeCommand()







def myname():
    print("My friends call me "+ assname+" , I am your virtual assistant created by Don exe ")
    speak("My friends call me "+ assname+" , I am your virtual assistant created by Don exe ")
    

def Quit():

    speak('bye sir')
    sys.exit()

def howClient():
    choicefile=open("username.json","r")
    linelist=[]
    for line in choicefile:
        linelist.append(line)
    choice=random.choice(linelist)
    speak(choice)

def reason():
    choicefile=open("texts/creator.txt","r")
    linelist=[]
    for line in choicefile:
        linelist.append(line)
    choice=random.choice(linelist)
    speak(choice)

def Whatislove():
    speak("bro . my creator doesn't like these topics ,  so ask about somthig else , maybe some music ,  my creator has some great songs")

def LockWindow():
    speak("locking the device")
    ctypes.windll.user32.LockWorkStation()

def Shutdown():
    speak("Hold On a Sec ! Your system is on its way to shut down")
    subprocess.call('shutdown / p /f')

def emptRB():
    winshell.recycle_bin().empty(confirm = False, show_progress = False, sound = True)
    speak("Recycle Bin Recycled")

def stplistent():
    order = takeCommand()
    if order == 'elliot' or order=='Elliot':
        speak("yes sir")
    else : 
        stplistent()


def locationSearch():
    query = query.replace("where is", "")
    location = query
    speak("User asked to Locate")
    speak(location)
    webbrowser.open("https://www.google.nl / maps / place/" + location + "")

def cam():
    ec.capture(0, " Camera ", "img.jpg")

def Restart():
    subprocess.call(["shutdown", "/r"])

def hibernate():
    speak("Hibernating")
    subprocess.call("shutdown / h")

def logOFF():
    speak("Make sure all the application are closed before sign-out")
    time.sleep(5)
    subprocess.call(["shutdown", "/l"])

def WAnote():
    now = datetime.now()
    speak("ok , What should i write")
    note = takeCommand()
    file = open('texts/note.txt', 'w')
    timeNow = now.strftime("%m/%d/%Y")
    noteTime = timeNow
    print(timeNow)
    file.write(noteTime)
    file.write(" :-. ")
    file.write(note)


def SNote():
    file = open("texts/note.txt", "r")
    file = file.read()
    print(file)
    speak(file)   

def oppwiki():
    webbrowser.open("https://www.wikipedia.com")




def NOlgbtg():
    speak('im a straight AI i dont support LGBTQ community ')
    print('im a straight AI i dont support LGBTQ community ')






#----------------END_BRAIN_ELLIOT---------------------------------




#------------------openAI-------------image-GEN--------------

doc = docx.Document()


def openFillings(data):
  openai.api_key = "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"


  response = openai.Completion.create(
    model="text-davinci-003",
    prompt=data,
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )

  txtcall =(response['choices'][0]['text'])
  list2 = txtcall.replace('\n', ' ')


  speak(list2)
  print(list2)

 


#this feature is under building
def imgGen(data):
  data = data
  openai.api_key = "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"
  data = data

  response = openai.Image.create(
    prompt=data,
    n=3,
    size="1024x1024"
  )
  url_1 = response['data'][0]['url']
  url_2 = response['data'][1]['url']
  url_3 = response['data'][2]['url']


  print(response)
def beta_imgGen(data):

    speak('please select folder to save images')
    destDir=filedialog.askdirectory()
    data=data
    generator = Craiyon() # Instantiates the api wrapper
    speak(f' Genetating {data} images. please wait')
    result = generator.generate(data) # Generates 9 images by default and you cannot change that
    result.save_images()
    sourceDir = './generated'
    
    
    moveAllFilesinDir(sourceDir,destDir)
    speak('the images generated succesfully')


#♀---------------DEEP WORK /// beta--------------------------------------

def adv_ImgGen(txt):
  r = requests.post(
    "https://api.deepai.org/api/fantasy-world-generator",
    data={
        'text': txt,
    },
    headers={'api-key': '2c370ab7-e1d9-4b2c-b02b-e7457358afca'}
  )
  print(r.json())


def TextGen(data):


    openai.api_key = "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"

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=data,
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    speak('please select folder to save text')

    destDir=filedialog.askdirectory()

    txtcall =(response['choices'][0]['text'])
    list2 = txtcall.replace('\n', ' ')
    file = open(f'{destDir}/generated text.txt', 'w')
    file.write(list2)
    sourceDir = '/generated'
    moveAllFilesinDir(sourceDir,destDir)


    speak('the text generated succesfully')

def DocGen(data):
    openai.api_key = "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"

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=data,
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )

    txtcall =(response['choices'][0]['text'])
    doc_para = doc.add_paragraph(txtcall)

    speak('what should i name the file ??')
    name = takeCommand()
    file = open(f'documents/{name}.doc', 'w')
    doc.save(f'documents/{name}.doc')
    destDir=filedialog.askdirectory()   
    moveAllFilesinDir(sourceDir,destDir)

    speak('the document  generated succesfully')

#--------------------OPEN-AI--END-------------------

def callmewhenyouwant():
    if __name__ == '__main__':
        clear = lambda: os.system('cls')

        if uname == "" :

            clear()
            intro()
            usname_set()
            username()

            
        else :
            clear()
            wishMe()

    while True:
            
            #query = input('---->')
            query =takeCommand()

            if 'play music' in query or "play song" in query:
                playmusic()
            elif 'the time' in query:
                thetime() 
            elif 'how are you' in query or 'are you ok' in query:
                howme()
            
            elif 'open brave' in query:
                brave()
        
            elif "change my name" in query:
                speak('what should i call you ?')
                NewName= takeCommand()
                with open("/username.json", "w") as f:
                    json.dump(""+NewName,f)
                speak(f'so welcome {NewName}')
            elif "what's your name" in query or "what is your name" in query:
                myname()
            elif 'exit' in query or "close" in query or "bye" in query or "goodbye" in query:
                Quit()
            elif "who made you" in query or "who created you" in query:
                creator()
            elif 'joke' in query:
                joke()
            elif "who i am" in query:
                howClient()
            elif "why you came to world" in query:
                reason()
            elif "weather" in query:
                weather()
            elif 'power point presentation' in query:
                speak("opening Power Point presentation")
                power = r"C:\\Users\\GAURAV\\Desktop\\Minor Project\\Presentation\\Voice Assistant.pptx"
                os.startfile(power)
            elif 'is love' in query:
                Whatislove()
            elif "who are you" in query:
                myname()
            elif 'reason for you' in query:
                reason()
            elif 'lock window' in query:
                    LockWindow()

            elif 'shutdown system' in query:
                Shutdown()
                    
            elif 'search' in query or 'play' in query:
                song= query.replace('play','')

                pywhatkit.playonyt(song)


            elif 'empty recycle bin' in query:
                emptRB()

            elif "don't listen" in query or "stop listening" in query:
                stplistent()
            elif "where is" in query:
                locationSearch()

            elif "camera" in query or "take a photo" in query:
                cam()
            elif "restart" in query:
                Restart()
                
            elif "hibernate" in query or "sleep" in query:
                hibernate()

            elif "log off" in query or "sign out" in query:
                logOFF()

            elif "write a note" in query:
                WAnote()
            
            elif "show note" in query:
                SNote()

            elif "elliot" in query or "Elliot" in query:
                speak("yes sir.")
                

            elif "open wikipedia" in query:
                oppwiki()

            elif "Good Morning" in query:
                speak("A warm" +query, "How are you  "+uname )
                

            # most asked question from google Assistant
            elif "will you be my gf" in query or "will you be my bf" in query:  
                speak("I'm not sure about, may be you should give me some time")

            elif "i love you" in query:
                speak("It's hard to understand")

            elif "what is" in query or "who is" in query:
                if query=="what is your name":

                    myname()
                else:

                    # Use the same API key
                    # that we have generated earlier
                    client = wolframalpha.Client("6AXTYW-KK7XPTUYY5")
                    res = client.query(query)
                    
                    try:
                        print (next(res.results).text)
                        speak (next(res.results).text)
                    except StopIteration:
                        print ("No results")



            elif 'open youtube' in query:
                speak("Here you go to Youtube")
                webbrowser.open("https://www.youtube.com")

            elif 'open facebook' in query:
                speak("lets talk to world in facebook")
                webbrowser.open("https://www.facebook.com")

            elif 'open soundcloud' in query:
                speak("lets vibe in soundcloud")
                webbrowser.open("https://www.soundcloud.com")

            elif 'open google' in query:
                speak("Here you go to Google")
                webbrowser.open("https://www.google.com")

            elif 'open stackoverflow' in query:
                speak("Here you go to Stack Over flow.Happy coding")
                webbrowser.open("https://www.stackoverflow.com")  
            elif 'news' in query:
                        
                        try:
                            jsonObj = urlopen('''https://newsapi.org / v1 / articles?source = the-times-of-india&sortBy = top&apiKey =\\times of India Api key\\''')
                            data = json.load(jsonObj)
                            i = 1
                            
                            speak('here are some top news from the times of india')
                            print('''=============== TIMES OF INDIA ============'''+ '\n')
                            
                            for item in data['articles']:
                                
                                print(str(i) + '. ' + item['title'] + '\n')
                                print(item['description'] + '\n')
                                speak(str(i) + '. ' + item['title'] + '\n')
                                i += 1
                        except Exception as e:
                            
                            print(str(e))
            elif "generate image" in query :
                speak("what image you want me generate ?")
                generattxt = takeCommand()
                beta_imgGen(generattxt)
            elif 'lgbtq' in query : 
                NOlgbtg()
            elif 'generate text' in query:
                
                TextGen(query)
            elif 'generate document' in query:
                speak("about what?")
                about = takeCommand()
                DocGen(about)
            else :
                openFillings(query)
                    




#-----------------------GUI----------------------------------------


root = tk.Tk()
root.title("Elliot AI")
root.geometry("920x670+290+85")
root.configure(bg="#0f1a2b")
root.resizable(False,False)

index =0

def helloCallBack():
    callmewhenyouwant()





    
#icon
image_icon = PhotoImage(file="img/logo.png")
root.iconphoto(False,image_icon)

top =PhotoImage(file="img/top3.png")
Label(root,image=top,bg="#0f1a2b").pack()

#logo
logo = PhotoImage(file='img/1.png')
Label(root,image=logo, bg="#0f1a2b").place(x=250,y=100)










#button
play_button = PhotoImage(file='img/start.png')
Button(root,image=play_button,bg="#0f1a2b",bd=0,command=helloCallBack).place(x=350,y=450)










root.mainloop()

#---------------------END_GUI--------------------------------------
