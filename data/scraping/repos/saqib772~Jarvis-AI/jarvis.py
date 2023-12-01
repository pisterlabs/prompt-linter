import openai
import pyttsx3
import speech_recognition as sr
import webbrowser

# Set up OpenAI API key
openai.api_key = "####"

# Initialize text-to-speech engine
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

# Initialize speech recognition engine
r = sr.Recognizer()

# Initialize OpenAI completion object
completion = openai.Completion()

def Reply(question):
    prompt = f'Alpha: {question}\n AI Tell Me'
    response = completion.create(prompt=prompt, engine="text-davinci-002", max_tokens=1024)
    answer = response.choices[0].text.strip()
    return answer 

def speak(text):
    engine.say(text)
    engine.runAndWait()

def takeCommand():
    with sr.Microphone() as source:
        print('Listening....')
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        print("Recognizing.....")
        query = r.recognize_google(audio, language='en-in')
        print(" Alpha Said: {} \n".format(query))
    except sr.UnknownValueError:
        print("Sorry, I didn't understand that.")
        return ""
    except sr.RequestError:
        print("Sorry, my speech service is not available right now.")
        return ""
    return query.lower()

if __name__ == '__main__':
    while True:
        query = takeCommand()
        if not query:
            continue
        ans = Reply(query)
        print(ans)
        speak(ans)
        
        if 'open youtube' in query:
            webbrowser.open("www.youtube.com")
        elif 'open google' in query:
            webbrowser.open("www.google.com")
        elif 'bye' in query:
            break
        else:
            print("Sorry, I'm not sure how to help with that.")
            speak("Sorry, I'm not sure how to help with that.")
