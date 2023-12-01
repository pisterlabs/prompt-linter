import os
import io
import json
import requests as r
from dotenv import load_dotenv

import speech_recognition as sr

from openai import OpenAI

from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import LanguageTranslatorV3

from pydub import AudioSegment
from pydub.playback import play

load_dotenv()

client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")

ibm_api_key = os.getenv("IBM_API_KEY")
ibm_tts_api_key = os.getenv("IBM_TTS_API_KEY")
ibm_tts_ibm_url = os.getenv("IBM_TTS_URL")
ibm_translate_api_key = os.getenv("IBM_TRANSLATE_API_KEY")
ibm_translate_url = os.getenv("IBM_TRANSLATE_URL")

authenticator = IAMAuthenticator(ibm_api_key)
text_to_speech = TextToSpeechV1(authenticator=authenticator)
text_to_speech.set_service_url(ibm_tts_ibm_url)


def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        said = ""

        try:
            said = r.recognize_google(audio, language="es-ES")
        except Exception as e:
            print("Exception: ", str(e))
    return said


system_roles = {
    "spanish_teacher": "You are a Spanish teacher. You will help me learn the basic of Spanish.",
}

messages = [
    {
        "role": "system",
        "content": system_roles["spanish_teacher"],
    }
]


def get_chat_response():
    print("getting chat response...")
    try:
        chat_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1024,
            temperature=0.9,
        )
        chat_response_content = chat_response.choices[0].message.content
        return {"text": chat_response_content}
    except Exception as e:
        print("Exception: ", str(e))


spanish_female_voice = "es-ES_LauraV3Voice"


def get_tts_response(chat_response_content):
    print("getting tts response...")
    try:
        headers = {
            "Content-Type": "application/json",
            "Accept": "audio/wav",
        }
        tts_post_url = f"/v1/synthesize?voice={spanish_female_voice}"
        tts_response = r.post(
            ibm_tts_ibm_url + tts_post_url,
            auth=("apikey", ibm_tts_api_key),
            headers=headers,
            data=json.dumps(chat_response_content),
        )

        audio_data = tts_response.content
        audio_segment = AudioSegment.from_wav(io.BytesIO(audio_data))
        play(audio_segment)
        return audio_data
    except Exception as e:
        print("Exception: ", str(e))


def english_translation(chat_response_content):
    print("translating to English...")
    authenticator = IAMAuthenticator(ibm_translate_api_key)
    language_translator = LanguageTranslatorV3(
        version="2018-05-01",
        authenticator=authenticator,
    )
    language_translator.set_service_url(ibm_translate_url)
    translation = language_translator.translate(
        text=chat_response_content["text"],
        model_id="es-en",
    ).get_result()
    eng_translation = translation["translations"][0]["translation"]
    print(
        json.dumps(
            eng_translation,
            indent=2,
            ensure_ascii=False,
        )
    )
    return eng_translation


while True:
    input("Press Enter to start recording...")
    audio_data = get_audio()
    print("user said: ", audio_data)
    messages.append(
        {
            "role": "user",
            "content": audio_data,
        }
    )
    chat_response_content = get_chat_response()
    messages.append(
        {
            "role": "assistant",
            "content": chat_response_content["text"],
        }
    )
    print("chat response content", chat_response_content)
    tts_response = get_tts_response(chat_response_content)
    eng_translation = english_translation(chat_response_content)
