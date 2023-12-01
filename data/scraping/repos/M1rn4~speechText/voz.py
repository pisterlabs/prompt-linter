import openai
from dotenv import load_dotenv
import os


# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Obtener la clave de API desde las variables de entorno
api_key = os.getenv("OPENAI_API_KEY")
print(api_key)
# Configurar la clave de API en openai
openai.api_key = api_key

# Abrir el archivo de audio
with open("./audio.mp3", "rb") as audio_file:
    # Transcribir el audio
    response = openai.Transcription.create(audio=audio_file)
    transcript = response['transcriptions'][0]['text']

    # Imprimir la transcripción
    print(transcript)

# Generar la voz del texto traducido
respuesta = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": transcript}
    ]
)

# Obtener la respuesta generada
voz_generada = respuesta.choices[0].message.content.strip()

# Imprimir la voz generada
print(voz_generada)
