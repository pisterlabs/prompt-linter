# GOOGLE SPEECH RECO MODEL + OPENAI BASE AI
import pyttsx3
import openai
# import opencv 
# import mediapipe
import speech_recognition as sr
from time import sleep
# replace sr by whisper

def talk(): #using pyttsx3
    words = input() 
    engine = pyttsx3.init() 
    engine.setProperty('rate',125)  
    engine.setProperty('volume',2.0)   
    voices = engine.getProperty('voices')       
    engine.setProperty('voice', voices[0].id) 
    engine.say(words)
    engine.runAndWait()
    engine.stop()
    
def talk(word):
    engine = pyttsx3.init() 
    engine.setProperty('rate',125)  
    engine.setProperty('volume',2.0)   
    voices = engine.getProperty('voices')     
    engine.setProperty('voice', voices[0].id) 
    engine.say(word)
    engine.runAndWait()
    engine.stop()
    
def myCommand(): 
    r = sr.Recognizer()                                                                                    
    with sr.Microphone() as source: 
        talk("Next command sir")  
        sleep(.25)                                                                     
        print("I am listening:") 
        r.pause_thresholdld =  1 
        audio = r.listen(source) 
    try: 
        query = r.recognize_google(audio, language='en-in') 
        return query 
    
    except sr.UnknownValueError: 
        print("Sorry. I didn't get that")
        talk("Sorry. I didn't get that")
        sleep(1)
        return "rerun"

while True:
    f = open("C:\\Users\\Home\\Desktop\\Programs\\keyOpenAI.txt",'r')
    key = f.read()
    f.close()
    openai.api_key = key
    myPrompt = myCommand()  
    if myPrompt == "rerun":
        continue
    
    elif myPrompt == "stop":
        print("Stopping program...")
        sleep(.5)
        talk("Goodbye")
        break
    
    else:
        reply = openai.Completion.create(engine = 'text-davinci-003', prompt = myPrompt,max_tokens = 50,temperature = .7)
        print(reply["choices"][0]["text"])
        talk(reply["choices"][0]["text"])
