import openai
import speech_recognition as sr
from gtts import gTTS
import os
from homeassistant_api import HomeAssistantAPI

# Imposta la chiave API di OpenAI
openai.api_key = 'TUA_CHIAVE_API_OPENAI'

# Imposta i dettagli del server Home Assistant
HA_HOST = 'indirizzo_del_tuo_server'
HA_API_KEY = 'TUA_CHIAVE_API_HOME_ASSISTANT'

# Funzione per riconoscere il comando vocale con attivazione vocale
def recognize_speech(keyword):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Ascolto...")
        audio = recognizer.listen(source)

    try:
        # Trascrivi ciò che è stato detto
        text = recognizer.recognize_google(audio)
        print("Hai detto:", text)
        
        if keyword in text:
            return text
        else:
            return None
    except sr.UnknownValueError:
        return None

# Funzione per generare una risposta con GPT-3
def generate_response(text):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=text,
        max_tokens=50
    )
    return response.choices[0].text

# Funzione per inviare un comando a Home Assistant
def send_command_to_ha(command):
    api = HomeAssistantAPI(HA_HOST, HA_API_KEY)
    api.services.call("homeassistant", command)

# Funzione per generare un'immagine con DALL·E
def generate_image_with_dalle(description):
    response = openai.Image.create(
        model="image-alpha-001",
        description=description,
        size="256x256"
    )
    return response.assets[0].url

# Parola chiave per l'attivazione vocale (sostituita con "jarvis")
keyword = "omara"

while True:
    command = recognize_speech(keyword)
    
    if command is not None:
        if "accendi luci" in command:
            send_command_to_ha("light.turn_on")
            response = "Ho acceso le luci per te."
        else:
            response = generate_response(command)
            image_url = generate_image_with_dalle(command)
            print("URL dell'immagine generata:", image_url)
        print("Risposta:", response)
        tts = gTTS(response)
        tts.save("response.mp3")
        os.system("mpg321 response.mp3")
