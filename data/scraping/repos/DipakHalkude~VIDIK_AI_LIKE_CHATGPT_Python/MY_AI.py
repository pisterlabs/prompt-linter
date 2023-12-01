import speech_recognition as sr
import os
import pyttsx3
import webbrowser
import openai
import time
import random

chatstr=""
def chat(query):
    global chatstr
    openai.api_key =""                    #"WRITE_API_KEY_OF_OPENAI"
    chatstr+=f"Dipak: {query}\n Vidik: "
    # Rest of your code
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": chatstr,
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    try:
     say(response["choices"][0]["message"]["content"])
     chatstr+=f"{response['choices'][0]['message']['content']}\n"
     return response["choices"][0]["message"]["content"]

     f=open(f"Openai/{''.join(prompt.split('intelligence')[1:]).strip()}.txt","w")
     f.write(text)

    except:
        return("Some Error Occured")






def say(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def ai(prompt):
    openai.api_key = "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"
    text='f"OpenAI response for Prompt: {prompt}\n ********************\n\n"'

    # Rest of your code
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    try:
     print(response["choices"][0]["message"]["content"])
     text+=response["choices"][0]["message"]["content"]
     if not os.path.exists("Openai"):
            os.mkdir("Openai")

     f=open(f"Openai/{''.join(prompt.split('intelligence')[1:]).strip()}.txt","w")
     f.write(text)

    except:
        return("Some Error Occured")
    

def takeCommand():
    r = sr.Recognizer()  # Corrected typo here
    with sr.Microphone() as source:
        # r.pause_threshold = 1
        audio = r.listen(source)
        try:
            print("Recongnizing")
            query = r.recognize_google(audio, language="en-in")
            print(f"User said: {query}")
            return query
        except:
            return "Some Error occured sorry from Vidik"


if __name__ == '__main__':
    print("VS CODE")
    say("Hello, I am Vidik AI")
    while True:
        print("Listening....")
        query = takeCommand()
        sites=[["youtube","https://www.youtube.com"],["Wikipedia","https://www.wikipedia.com"],["google","https://www.google.com"],["instagram","https://www.instagram.com/"],['facebook',"https://www.facebook.com/"],['twitter',"https://www.twitter.com/"],]
        for site in sites:
            if f"Open {site[0]}".lower() in query.lower():
                webbrowser.open(site[1])
                say(f"Opening {site[0]} sir")
        
        if "postman".lower() in query.lower():
            path = r"\Users\dipak\OneDrive\Documents\Desktop\Postman.lnk"
            if os.name == 'nt':
               os.startfile(path)
  
        if "open music".lower() in query.lower():
            musicpath=r"\Users\dipak\OneDrive\Documents\Desktop\Spotify\songs\1.mp3"
            if os.name == 'nt':
               os.startfile(musicpath)
   
        if "the time".lower() in query.lower():
            nowtime=time.strftime("%H:%M:%S")
            print(nowtime)
            say(f" Sir the time is {nowtime}")

        if "Using artificial intelligence".lower() in query.lower():
            ai(prompt=query)

        elif "Goodbye".lower() in query.lower():
            say("Goodbye Sir,Thanks for using me")
            exit(0)
        
        elif "chat reset".lower() in query.lower():
            chatstr=""

        else:
            print("Chatting....")
            chat(query)
       
        
        # say(query)
