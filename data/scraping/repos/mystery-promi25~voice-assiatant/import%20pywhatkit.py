import struct
import pyaudio
import pvporcupine
import speech_recognition as sr
from gtts import gTTS 
import pywhatkit
import datetime
import wikipedia
import pyjokes
import random
from playsound import playsound
import openai
import shelve

porcupine = None
pa = None
audio_stream = None

hi_phases = ['how can i help you?', 'Hi', "what do you want", 'terminator, at your service']

listener = sr.Recognizer()
openai.api_key = "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"

def update_context(context, value):
    f = open(context+".txt", "a")
    f.write(value)
    f.close()
def run_alexa(command):
    print(command)
    if 'play' in command:
        song = command.replace('play', '')
        talk('playing ' + song)
        pywhatkit.playonyt(song)
    elif 'time' in command:
        time = datetime.datetime.now().strftime('%I:%M %p')
        talk('Current time is ' + time)
    elif 'who is' in command or 'what is' in command:
        person = command.replace('who the heck is', '')
        info = wikipedia.summary(person, 1)
        print(info)
        talk(info)
    elif 'date' in command:
        talk('sorry, I have a headache')
    elif 'are you single' in command:
        talk('I am in a relationship with wifi')
    elif 'joke' in command:
        talk(pyjokes.get_joke())
    elif 'chat' in command:
        talk('do you want me to load, create or not use a context file?')
        x = take_command()
        if 'load' in x:
            talk('please say the context name')
            context_name = take_command()
            c = True
        elif 'create' in x or 'make' in x:
            talk('what will the new context be called')
            context_name = take_command()
            c = True
        else:
            talk('what do you want me to ask chat-gpt?')
            question = take_command()
            response = openai.Completion.create(
            model="text-davinci-003",
            prompt=question,
            temperature=0.7,
            max_tokens=64,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
            )
            talk(response.choices[0].text)
        if c:
            while True:
                talk('what do you want me to ask chat-gpt? say exit if you want to return to the menu')
                question = take_command()
                if question == "exit":
                    break
                update_context(context_name,"\n\nUser: "+question)
                response = openai.Completion.create(
                model="text-davinci-003",
                prompt=question,
                temperature=0.7,
                max_tokens=64,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
                )
                talk(response.choices[0].text)
                update_context(context_name,"\n\nAI: "+response.choices[0].text)
    else:
        talk('Sorry i didn\'t get that')

def talk(text):
    myobj = gTTS(text=text, lang='en', slow=False) 
    myobj.save("welcome.mp3") 
    playsound("welcome.mp3")

def take_command():
    try:
        with sr.Microphone() as source:
            print('listening...')
            voice = listener.listen(source)
            command = listener.recognize_google(voice)
            command = command.lower()
    except:
        pass
    return command

while True:
    try:
        porcupine = pvporcupine.create(keywords=["terminator"], access_key='YmJlEZWXl9WGVV3qEJCA2TTdNdUiKqeyVrCn2dDEJARVTWyUtvfQWA==')
        pa = pyaudio.PyAudio()
        audio_stream = pa.open(
                        rate=porcupine.sample_rate,
                        channels=1,
                        format=pyaudio.paInt16,
                        input=True,
                        frames_per_buffer=porcupine.frame_length)
        while True:
            pcm = audio_stream.read(porcupine.frame_length)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            keyword_index = porcupine.process(pcm)
            if keyword_index >= 0:
                talk(random.choice(hi_phases))
                command = take_command()
                run_alexa(command)

    finally:
        if porcupine is not None:
            porcupine.delete()
        if audio_stream is not None:
            audio_stream.close()
        if pa is not None:
                pa.terminate()