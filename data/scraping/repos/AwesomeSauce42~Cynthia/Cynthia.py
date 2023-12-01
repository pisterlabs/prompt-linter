import openai
import pyttsx3
import speech_recognition as sr
import webbrowser

openai.api_key = "sk-...your secret key..."

completion = openai.Completion()

def Reply(text):
    prompt = f'Shash: {text}\n Cynthia:'
    response = completion.create(prompt=prompt, engine = "text-curie-001")
    answer= response.choices[0].text.rstrip()
    print
    return answer

engine=pyttsx3.init('sapi5')
voices=engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

def speak(text):
    engine.say(text)
    engine.runAndWait()

speak("Hello,I am Cynthia. How may I help you? ")

def takeCommand():
    r=sr.Recognizer()
    with sr.Microphone() as source:
        print('Listening....')
        r.pause_threshold = 0.5
        audio = r.listen(source)
    try:
        print("Recognizing.....")
        query=r.recognize_google(audio, language='en-in')
        print("Shash: {} \n".format(query))
    except Exception as e:
        speak("Could you repeat that?")
        return takeCommand()
    return query

if __name__ == '__main__':
    while True:
        query=takeCommand().lower()
        ans=Reply(query)
        print(ans)
        speak(ans)
        if 'open' in query:
            if 'youtube' in query: 
                webbrowser.open("www.youtube.com")
            if 'google' in query:
                webbrowser.open("www.google.com")
            else:
                webbrowser.open(query)
        if 'bye' in query:
            break
