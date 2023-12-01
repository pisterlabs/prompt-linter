import os
import openai
import pyttsx3
from gtts import gTTS
import speech_recognition as sr
import YanAPI

class GPT:
    def __init__(self,key,lang="en-US"):
        openai.api_key = key
        self.lang = lang
        
    def speech_to_text(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Nói gì đi...")
            audio = r.listen(source)
        try:
            text = r.recognize_google(audio,language=self.lang)
            print("You said: ", text)
            return text
        except: 
            return ""

    def text_to_speech(self,text):
        
        tts = gTTS(text=text, lang=self.lang)
        tts.save('output.mp3')
        YanAPI.upload_media_music('output.mp3')
        YanAPI.sync_play_music('output.mp3')
        YanAPI.delete_media_music('output.mp3')
        
        

    def ask_gpt(self,question):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=question,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
            timeout=5

        )
        answer = response.choices[0].text.strip()
        return answer
    
    def Run(self):
        with open('GioiThieu.txt','r') as f:
            data= f.read()
        self.text_to_speech(data)
key = '1'
gpt = GPT(key,lang="vi") #,lang="en-US"s
gpt.Run()
