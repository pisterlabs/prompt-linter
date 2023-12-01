# Libraries must be imported from the console or from the command line by running pip install <library>
import speech_recognition as sr
import openai
from gtts import gTTS
from pygame import mixer
import pyttsx3 # If the gTTS library does not work, you can use this one, which is the most used
import os
import time as ti
import random


openai.api_key = "Paste your API KEY here"


# We define the function that transforms the voice captured in the mic to text
def transform_audio_in_text():

    r = sr.Recognizer()
    with sr.Microphone() as origen:
        r.pause_threshold = 0.9
        print("Speak something please!")
        audio = r.listen(origen)
        try:
            # The language with which the voice is recognized must be specified
            request = r.recognize_google(audio, language="en-EN") 
            print("You: " + request)
            return request

        except sr.UnknownValueError:
            print("Ups, Sorry I didn't understand!")
            return "Please can repeat?"

        except sr.RequestError:
            print("Ups, no service!")
            return "waiting"

        except:
            print("Ups, something are wrong!")
            return "waiting"

        
# We define the function that will transform the text (message) into audio, I leave both for gTTS library and for pyttsx3
def speak (mensaje):
#This section of code is for gTTS library **
    volume = 0.7
    tts = gTTS(mensaje, lang="en", slow=False)
    ran = random.randint(0,9999)
    filename = 'Temp' + format(ran) + '.mp3'
    tts.save(filename)
    mixer.init()
    mixer.music.load(filename)
    mixer.music.set_volume(volume)
    mixer.music.play()

    while mixer.music.get_busy():
        ti.sleep(0.3)

    mixer.quit()
    os.remove(filename)
    
  # ******* This section of code is for pyttsx3 library, just use one removing the numeral 
    # In case the gTTS library doesn't work
    # Start the pyttsx3 engine
    #engine = pyttsx3.init()
    # Slow down playback speed, default is 200
    #engine.setProperty('rate', 150)
    # speak message
    #engine.say(message)
    # engine.runAndWait()

def despedida():
    speak("See you, have a great day!")
    exit()

def main():
    conversation = ""
    despedida_phrases = ["tchau", "thank you by", "see you later", "see you soom", "i have to go", "end", "see off"]
    speak ("Hello I'm Julie, welcome, I'm willing to answer all of your questions, please let's start the conversation!")

    while True:
        question = transform_audio_in_text().lower()

        conversation += "\nYou: " + question + "\nJulie:"
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=conversation,
            temperature=0.5,
            max_tokens=3500,
            top_p=0.5,
            frequency_penalty=0.7,
            presence_penalty=0.0,
            stop=["\n", " You:", " Julie:"]
        )
        answer = response.choices[0].text.strip()
        conversation += answer
        print("Julie: " + answer + "\n")
        speak(answer)
         # Check if any of the despedida phrases was said to stop the conversation
        if any(frase in question for frase in despedida_phrases):
            despedida()
main()