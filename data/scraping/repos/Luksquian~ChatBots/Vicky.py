# *****************************************************************************************************
# *****************************************************************************************************
# *************************** CODIGO CREADO Y COMPARTIDO POR LKTECHAI *********************************
# *****************************************************************************************************
# *****************************************************************************************************

# Las librerias deben importarse desde la consola o desde linea de comnado ejecutando pip install <libreria>
import speech_recognition as sr
import openai
from gtts import gTTS
from pygame import mixer
import pyttsx3 # Si la libreria gTTS no funciona se puede utilizar esta que es la mas utilizada
import os
import time as ti
import random


openai.api_key = "<YOUR APY KEY FROM OPENAI ACCOUNT>"


# Definimos la funcion que transforma la voz captada en el mic a texto
def transformar_audio_a_texto():

    r = sr.Recognizer()
    with sr.Microphone() as origen:
        r.pause_threshold = 0.8
        print("Ya puedes hablar!")
        audio = r.listen(origen)
        try:
            pedido = r.recognize_google(audio, language="es-AR") # Se debe especificar el lenguaje con el que se reconoce la voz
            print("You: " + pedido)
            return pedido

        except sr.UnknownValueError:
            print("Ups, no entendi!")
            return "Sigo esperando"

        except sr.RequestError:
            print("Ups, no hay servicio!")
            return "Sigo esperando"

        except:
            print("Ups, algo salio mal!")
            return "Sigo esperando"

        
# Definimos la funcion que va a transformar el texto (mensaje) en audio, dejo tanto para libreria gTTS como para pyttsx3
def hablar(mensaje):
    # ******* Esta seccion de codigo es para libreria gTTS ********
    volume = 0.7
    tts = gTTS(mensaje, lang="es", slow=False)
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
    
    # ******* Esta seccion de codigo es para libreria pyttsx3, solo utilizar una quitando el numeral ********
    # En caso de que la libreria gTTS no funcione 
    # Encender el motor de pyttsx3
    #engine = pyttsx3.init()
    # Bajar velocidad de reproduccion, por defecto es 200
    #engine.setProperty('rate', 150)

    # Pronunciar mensaje
    #engine.say(mensaje)
    #engine.runAndWait()

def main():
    conversation = ""

    hablar("Hola! Soy Vicky tu asistente personal, ¿en que puedo ayudarte?")

    while True:
        question = transformar_audio_a_texto().lower()

        conversation += "\nYou: " + question + "\nVicky:"
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=conversation,
            temperature=0.5,
            max_tokens=100,
            top_p=0.3,
            frequency_penalty=0.5,
            presence_penalty=0.0,
            stop=["\n", " You:", " Vicky:"]
        )
        answer = response.choices[0].text.strip()
        conversation += answer
        print("Vicky: " + answer + "\n")
        hablar(answer)


main()
