import speech_recognition
import pyttsx3
import pywhatkit
import datetime
import wikipedia
import openai

engine = pyttsx3.init()
openai.api_key = "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"
listener = speech_recognition.Recognizer()


voices = engine.getProperty('voices')      
engine.setProperty('voice', voices[1].id) 
engine.setProperty('rate', 125) 
engine.say('Hello, How can I help you?')
engine.runAndWait()


def reply_command(command):
    
    if 'play' in command:
        song = command.replace(command[:command.index('play') + 5], '')
        engine.say('Ok playing ' + song)
        pywhatkit.playonyt(song)
    elif 'time' in command:
        engine.say('Current time is ' + datetime.datetime.now().strftime('%H:%M'))
    elif 'search' in command:
        item = command.replace(command[: command.index('search') + 7], '')
        info = wikipedia.summary(item, 1)
        print(info)
        engine.say('Ok. ' + info)
    elif 'your name' in command:
        engine.say('My name is Alexa. Whats your name?')
    elif 'my name' in command:
        name = command.replace(command[: command.index('is') + 3], '')
        
        engine.say('Nice to meet you ' + name + '. How can I help you?')
    elif 'you single' in command:
        engine.say('No, currently I am in a relationship with Shahed')
    elif 'bye' in command:
        engine.say('Ok bye. See you soon!')
        # exit()
    else:
        print('GPT: ',command)
        response = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = [ {"role": "user", "content": command} ]
        )

        print(response)
        
        engine.say(response.choices[0].message.content)


    engine.runAndWait()


def take_command():
    try:
        with speech_recognition.Microphone() as source:
            listener.adjust_for_ambient_noise(source, duration=0.2)
            print('Listening...')
            audio = listener.listen(source)
            print(audio)
            command = listener.recognize_google(audio)
            command = command.lower()
            reply_command(command)
            
    except:
        print('Something went wrong.')
        # take_command()
        


while True:
    take_command()

