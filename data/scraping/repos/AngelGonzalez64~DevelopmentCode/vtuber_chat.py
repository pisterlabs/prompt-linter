# Para obtener credenciales en cada ejecucion...
#$env:GOOGLE_APPLICATION_CREDENTIALS="C:\Users\angel\OneDrive\Documentos\Codigos_Random\Chat\vtuberchat-384008-9a4c7fd95572.json"

import openai
import os
from google.cloud import texttospeech

#De aqui sacas el archivo Json que necesitas: https://console.cloud.google.com/
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/angel/OneDrive/Documentos/Codigos_Random/Chat/vtuberchat-384008-9a4c7fd95572.json"

# Reemplaza esto con tu clave de API de OpenAI
openai.api_key = "TuClaveDeAccesoDeOpenAIAqui"

def generate_text(prompt, prompt_lang="es", response_lang="es"):
    prompt = f"Una persona que habla español pregunta: '{prompt}'. Respuesta: {response_lang}"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.7,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return response.choices[0].text.strip()

def synthesize_speech(text):
    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
    language_code="es-ES",
    name="es-ES-Standard-A",  # Utiliza una voz femenina específica
    ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
)


    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )

    with open("output.mp3", "wb") as out:
        out.write(response.audio_content)
        print("Audio content written to file 'output.mp3'")

def chat_and_speak(question):
    response_text = generate_text(question)
    synthesize_speech(response_text)

if __name__ == "__main__":
    question = input("Please enter your question: ")
    chat_and_speak(question)
