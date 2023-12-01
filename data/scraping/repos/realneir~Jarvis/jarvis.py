import speech_recognition as sr
import pyttsx3
import openai
import requests
import webbrowser

engine = pyttsx3.init()

openai.api_key = 'YOUR_API_KEY'  

r = sr.Recognizer()

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

def get_audio():
    with sr.Microphone() as source:
        print("Identifying voice...")
        audio = r.listen(source)
        text = ""

        try:
            print("Recognizing...")
            text = r.recognize_google(audio)
            print(text)
        except Exception as e:
            print(e)
            speak_text("Sorry, Master Reneir can you please repeat.")

        return text

def get_response(prompt):
    if "open google" in prompt.lower():
        webbrowser.open("http://www.google.com")
        return "Opening Google"
    elif "open facebook" in prompt.lower():
        webbrowser.open("http://www.facebook.com")
        return "Opening Facebook"
    else:
        response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=100)
        return response.choices[0].text.strip()

speak_text("Hello, I am Jarvis from Ironman. How can I assist you today?")
while True:
    text = get_audio()

    if text:
        response = get_response(text)

        speak_text(response)
