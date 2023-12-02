import openai
import os
import speech_recognition as sr
from google.cloud import texttospeech
from pydub import AudioSegment
from pydub.playback import play

#De aqui sacas el archivo Json que necesitas: https://console.cloud.google.com/
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/angel/OneDrive/Documentos/Codigos_Random/Chat/vtuberchat-384008-9a4c7fd95572.json"

# Clave de API de OpenAI
openai.api_key = "TuClaveDeAccesoDeOpenAIAqui"


def generate_text(prompt, prompt_lang="es", response_lang="es"):
    prompt = f"Una persona que habla español pregunta: '{prompt}'. Por favor, responde de manera amigable y detallada en {response_lang}:"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.7,
        max_tokens=300,
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
        name="es-ES-Standard-A",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    response = client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )

    with open("output.wav", "wb") as out:
        out.write(response.audio_content)
        print("Audio content written to file 'output.wav'")


def play_audio(file_name):
    audio = AudioSegment.from_wav(file_name)
    play(audio)


def get_audio_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Por favor, haz tu pregunta:")
        audio = recognizer.listen(source)
        try:
            question = recognizer.recognize_google(audio, language="es-ES")
            print(f"Pregunta reconocida: {question}")
            return question
        except sr.UnknownValueError:
            print("Lo siento, no pude entender tu pregunta.")
            return None
        except sr.RequestError as e:
            print(
                f"Error al llamar al servicio de reconocimiento de voz de Google; {e}")
            return None


def chat_and_speak():
    question = get_audio_input()
    if question is not None:
        response_text = generate_text(question)
        synthesize_speech(response_text)
        play_audio("output.wav")

if __name__ == "__main__":
    chat_and_speak()