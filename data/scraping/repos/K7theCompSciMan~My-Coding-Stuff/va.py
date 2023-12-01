import speech_recognition as sr
import pyttsx3
import openai
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_APIKEY')
openai.api_key = OPENAI_API_KEY

def speak(text):
    engine = pyttsx3.init()

    # Adjust speech parameters
    engine.setProperty('rate', 200)  # Adjust the speaking rate (words per minute)
    engine.setProperty('volume', 0.8)  # Adjust the volume level

    # Retrieve available voices and select one
    voices = engine.getProperty('voices')
    for voice in voices:
        if "jarvis" in voice.name.lower():  # Find a voice resembling JARVIS
            engine.setProperty('voice', voice.id)
            break

    engine.say(text)
    engine.runAndWait()

def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.2)
        try:
            audio = recognizer.listen(source, timeout=3) 
        except:
            pass

    try:
        print("Recognizing...")
        statement = recognizer.recognize_google(audio)
        print(f"User said: {statement}\n")
        return statement.lower()
    except sr.UnknownValueError:
        speak("Sorry, I couldn't understand. Please say that again.")
    except sr.RequestError:
        speak("Speech recognition service is unavailable. Please try again later.")
    return ""

def send_to_chatGPT(messages, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.5
    )
    messages.append(response.choices[0].message)
    return response.choices[0].message['content']

messages = [{"role": "user", "content": "You are JARVIS from Iron Man. Do not break character in any way, even if it is to state your limitations as an AI language model. Respond to this with, 'Hello, I am JARVIS, how may I help you?'"}]
speak(send_to_chatGPT(messages))
exit_words = ["goodbye", "bye", "exit", "quit", "stop", "that's all", "that's it", "that is all", "that is it", "that's all for now", "that's it for now", "that is all for now", "that is it for now", "that's all, thanks", "that's it, thanks", "that is all, thanks", "that is it, thanks", "that's all for now, thanks", "that's it for now, thanks", "that is all for now, thanks", "that is it for now, thanks", "that's all, thank you", "that's it, thank you", "that is all, thank you", "that is it, thank you", "that's all for now, thank you", "that's it for now, thank you", "that is all for now, thank you", "that is it for now, thank you"]

def main():
    while True:
        text = listen()
        if text:            
            messages.append({"role": "user", "content": text})
            response = send_to_chatGPT(messages)
            for word in exit_words:
                if word in response:
                    speak("Goodbye")
                    exit()
            speak(response)
            print(response)

if __name__ == "__main__":
    main()