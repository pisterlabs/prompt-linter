import openai
from datetime import datetime
from logging import exception
from unicodedata import name
import pyttsx3
import speech_recognition as sr
import pyaudio
import wikipedia
import webbrowser
import os
import speech_recognition as sr
from gtts import gTTS
import json

openai.api_key = "Paste YOUR API key"

# Load previous conversations from a JSON file
def load_conversations():
    try:
        with open("conversations.json", "r") as f:
            return json.load(f)
    except:
        return []

# Save the current conversation to a JSON file
def save_conversations(conversations):
    with open("conversations.json", "w") as f:
        json.dump(conversations, f)

# Convert your voice input to text using SpeechRecognition
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def wishme():
    hour = int(datetime.now().hour)
    if hour >= 0 and hour < 12:
        speak("gaurav; good morning, have a nice day")
    elif hour >= 12 and hour < 18:
        speak("gaurav,good afternoon. spend your day efficiently")
    else:
        speak("good night how was your day")
    speak("it's jarvis, how may i help you")

# r = sr.Recognizer()
# with sr.Microphone() as source:
#     print("Speak your question...")
#     audio = r.listen(source)
#     question = r.recognize_google(audio)

def takecommand(conversations):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("i am listening and processing")
        r.pause_threshold = 1
        r.adjust_for_ambient_noise
        r.energy_threshold = 250
        audio = r.listen(source)

    try:
        print("recognizing ")
        speak(" just wait")
        query = r.recognize_google(audio, language='en-in')
        print(f"user said:{query}")
    except Exception as e:
        speak(" please say again")
        return "none"

    # Use previous conversations to provide more context
    prompt = "Q: " + query + "\nA:"
    for conversation in reversed(conversations):
        prompt += " " + conversation["question"] + "\nA: " + conversation["answer"] + "\n"

    return prompt

if __name__ == "__main__":
    prev_response = ""
    conversations = load_conversations()
    wishme()
    while True:
        query = takecommand(conversations).lower()
        model_engine = "text-davinci-003" # Or any other model of your choice
        response = openai.Completion.create(
            engine=model_engine,
            prompt=f"Q: {prev_response} A: {query}\nA:",
            max_tokens=2048,
            n=1,
            stop=None,
            temperature=1,
        )

        answer = response.choices[0].text.strip()
        print(answer)
        speak(answer)

        prev_response = answer

        # Add the current conversation to the list of previous conversations and save it to the JSON file
        conversations.append({"question": query, "answer": answer})
        save_conversations(conversations)

        if 'exit' in query:
            exit()


        

