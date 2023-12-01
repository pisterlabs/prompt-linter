import pyttsx3
import speech_recognition as sr
import datetime
import wikipedia
import webbrowser
import os
import subprocess
import smtplib
import requests
import json
import random
import openai
import credentials

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        return text
    except:
        speak("Sorry, I didn't understand that. Could you please repeat?")

def wish_me():
    hour = datetime.datetime.now().hour
    if hour >= 0 and hour < 12:
        speak("Good morning!")
    elif hour >= 12 and hour < 18:
        speak("Good afternoon!")
    else:
        speak("Good evening!")
    speak("How can I assist you today?")

def play_music():
    music_dir = 'C:\\Users\\User\\Music\\'
    songs = os.listdir(music_dir)
    os.startfile(os.path.join(music_dir, random.choice(songs)))

def send_email(recipient, subject, body):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login(credentials.EMAIL_USERNAME, credentials.EMAIL_PASSWORD)
    message = f'Subject: {subject}\n\n{body}'
    server.sendmail(credentials.EMAIL_USERNAME, recipient, message)
    server.quit()

def get_weather():
    api_key = credentials.WEATHER_API_KEY
    base_url = 'http://api.openweathermap.org/data/2.5/weather?'
    speak("Which city's weather would you like to know?")
    city = listen().lower()
    complete_url = f'{base_url}q={city}&appid={api_key}'
    response = requests.get(complete_url)
    data = response.json()
    if data['cod'] != '404':
        weather = data['weather'][0]['description']
        temperature = round(data['main']['temp'] - 273.15, 2)
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']
        speak(f"The temperature in {city} is {temperature} degrees Celsius. The weather is {weather}, with a humidity of {humidity}% and a wind speed of {wind_speed} meters per second.")
    else:
        speak("Sorry, I could not find the weather for that city.")

def get_news():
    api_key = credentials.NEWS_API_KEY
    url = f'http://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}'
    response = requests.get(url)
    data = json.loads(response.text)
    for index, article in enumerate(data['articles'][:5]):
        speak(f"{index+1}. {article['title']}")

def get_joke():
    joke = pyjokes.get_joke()
    speak(joke)

def get_answer(question):
    openai.api_key = credentials.OPENAI_API_KEY
    response = openai.Completion.create(
        engine="davinci",
        prompt=f"Q: {question}\nA:",
        temperature=0.7,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    answer = response.choices[0].text.strip()
    return answer

def assistant():
    speak("Initializing...")
    wish_me()
    while True:
        text = listen().lower()
        if "wikipedia" in text:
            speak("Searching Wikipedia...")
            query = text.replace("wikipedia", "")
            results = wikipedia.summary(query, sentences=3)
            speak("According to Wikipedia...")
            speak(results)
        elif "open youtube" in text:
            speak("Opening YouTube...")
            webbrowser.open("youtube.com")
        elif "open google" in text:
            speak("Opening Google...")
            webbrowser.open("google.com")
        elif "play music" in text:
            speak("Playing music...")
            play_music()
        elif "what's the weather" in text:
            get_weather()
        elif "tell me the news" in text:
            speak("Here are the top headlines for today...")
            get_news()
        elif "tell me a joke" in text:
            get_joke()
        elif "send email" in text:
            speak("To whom would you like to send the email?")
            recipient = listen()
            speak("What is the subject of your email?")
            subject = listen()
            speak("What would you like to say?")
            body = listen()
            send_email(recipient, subject, body)
        elif "what is" in text or "who is" in text or "where is" in text or "when is" in text:
            speak("Let me look that up for you...")
            answer = get_answer(text)
            speak(answer)
        elif "stop" in text or "exit" in text:
            speak("Goodbye!")
            break

if __name__ == '__main__':
    assistant()
