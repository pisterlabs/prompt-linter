from openai import OpenAI
import speech_recognition as sr
import requests
import json
from gtts import gTTS
import os


def speech_to_text():
    recognizer = sr.Recognizer()

    with sr.Microphone(device_index = 2, sample_rate = 48000) as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("請說話...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio, language="zh-TW")
        return text
    except sr.UnknownValueError:
        print("無法辨識語音")
        return None

def chat_with_gpt(text):
    gpt_api_url = "https://api.openai.com/v1/chat/completions"
    gpt_api_key = "..."

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {gpt_api_key}"
    }

    data = {
        "model": "gpt-3.5-turbo-1106",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant.。"},
            {"role": "user", "content": text}
        ]
    }

    response = requests.post(gpt_api_url, headers=headers, json=data)
    return response.json()["choices"][0]["message"]["content"]

if __name__ == "__main__":
    while True:
        user_input = speech_to_text()
        if user_input:
            gpt_response = chat_with_gpt(user_input)            
            print("ChatGPT:", gpt_response)
            tts = gTTS(text=gpt_response, lang='zh-TW')
            tts.save('tmp.mp3')
            os.system('omxplayer -o local -p tmp.mp3 > /dev/null 2>&1')
