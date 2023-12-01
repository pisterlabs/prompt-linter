import openai
import pyttsx3
import speech_recognition as sr
import webbrowser

from apikey import api_data

openai.api_key = api_data

completion = openai.Completion()

def generate_response(question):
    prompt = f'Tejas: {question}\n Jarvis: '
    response = completion.create(prompt=prompt, engine="text-davinci-002", stop=['\Tejas'])
    answer = response.choices[0].text.strip()
    return answer
    
ans = generate_response("What is openAI?")
print(ans)                                     

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)


def speak(text):
    engine.say(text)
    engine.runAndWait()
    
speak("Hello, How are you friend?")

def takecommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening....!")
        r.pause_threshold = 1
        audio = r.listen(source)
        
    try:
        print("Recognizing...")
        query = r.recognize_google(audio , language='en-in')
        print("Tejas said: {}\n".format(query))
        
    except Exception as e:
        print("Say that again...")
        return None
    return query

if __name__ == '__main__':
    while True:
        query = takecommand().lower()
        print(query)
        if 'goodbye' in  query:
            break 
