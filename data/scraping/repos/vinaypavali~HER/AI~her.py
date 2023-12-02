import openai
import speech_recognition as sr
import pyttsx3
from config import apikey

 
openai.api_key = apikey
 
engine = pyttsx3.init()

def get_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",   
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )
    return response['choices'][0]['text'].strip() # type: ignore

def takeCommand():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        user_input = recognizer.recognize_google(audio)
        print("Neelesh:", user_input)
        return user_input if user_input else ""
    except sr.UnknownValueError:
        print("Could not understand audio.")
    except sr.RequestError as e:
        print("Error accessing Google Speech Recognition service:", e)

    return ""

def speak(text):
    print("Samantha:", text)
    engine.say(text)
    engine.runAndWait()

def main():
    print("Samantha Voice Assistant")
    print("Say 'exit' to end the conversation.")

    while True:
        user_input = takeCommand()

        if user_input == 'exit':
            break

        if user_input:
            prompt = f"You said: {user_input}\nAI:"
            response = get_response(prompt)
            speak(response)

if __name__ == "__main__":
    main()
