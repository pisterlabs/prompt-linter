# Import libraries
import datetime
import webbrowser
import wikipedia

import speech_recognition as sr
import openai


class TalkGPT:
    def __init__(self, voice_engine, speech_recognizer):
        self.voice_engine = voice_engine
        self.speech_recognizer = speech_recognizer


    def speak(self, text_to_speech):
        self.voice_engine.say(text_to_speech)
        self.voice_engine.runAndWait()


    def listen(self):
        query = None
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            r.pause_threshold = 1.0
            audio = r.listen(source)
        try:
            print("Recognizing...")
            query = r.recognize_google(audio)
            print(f'Human said: "{query}"\n')
        except Exception as err:
            self.speak("Sorry, I did not catch what you said, please speak again")
            print(err)
        return query


    def activate(self):
        self.speak('Hello, I am TalkGPT. How can I help you?')
        return True


    def deactivate(self):
        self.speak('I hope I have helped you well. Good bye!')
        return False

    
    def simple_chat(self, sentence):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "user", "content": sentence}]
        )
        self.speak(response['choices'][0]['message']['content'])