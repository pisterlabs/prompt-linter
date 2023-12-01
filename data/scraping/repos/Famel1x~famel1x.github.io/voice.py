import openai
import pyttsx3
import speech_recognition
from googletrans import Translator

openai.api_key = "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"

r = speech_recognition.Recognizer()

tts = pyttsx3.init()
voices = tts.getProperty("voices")
tts.setProperty("voice", voices[0].id)

translator = Translator(service_urls=['translate.googleapis.com'])

record_result = ""
gpt_result = ""

def speech(text_to_speech):
    print("Ответ на фразу: " + text_to_speech)
    tts.say(text_to_speech)
    tts.runAndWait()

def askGPT(text):
    print("Мыслим....")
    textEn = translator.translate(text = text, dest = "en")

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": textEn.text}])

    answer = translator.translate(text = f"{completion.choices[0].message.content}", dest = "ru")
    global gpt_result
    gpt_result = answer.text
    return speech(answer.text)

def record():
    global record_result
    try:
        with speech_recognition.Microphone(device_index = 1) as sourse:
            audio = r.listen(sourse)
        record_voice_result = r.recognize_google(audio, language = "ru-RU").lower()
        record_result = record_voice_result
        print("Распознаная фраза: " + record_voice_result)
        askGPT(record_voice_result)
    except:
        record_result = "Ошибка записи"
        print(record_result)