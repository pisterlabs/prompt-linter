# %%
import speech_recognition as sr
import json
import openai
from gtts import gTTS
from bardapi import Bard
import os
import pygame

def recognize_speech_from_microphone():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("請開始說話...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio, language="zh-TW") # 將語音轉換為文字
            print("辨識結果：", text)
            return text
        except sr.UnknownValueError:
            print("無法辨識語音")
            return None
        except sr.RequestError:
            print("無法連接到Google API")
            return None

def generate_response_with_gpt(text):
    response = openai.Completion.create(
        model ="gpt-3.5-turbo-instruct",
        prompt = text,
        max_tokens = 500
    )
    response_text = response.choices[0].text.strip()
    print("GPT回應:", response_text)

    # 將GPT回應轉換為語音
    tts = gTTS(text = response_text, lang='zh-TW')
    tts.save("./Data/response.mp3")
    os.system("start ./Data/response.mp3")

def generate_response_with_bard(text):
    # https://zhuanlan.zhihu.com/p/631178245
    response_text = Bard().get_answer(text)["content"]
    print("BARD回應:", response_text)
    # 將BARD回應轉換為語音
    tts = gTTS(text = response_text, lang='zh-TW')
    tts.save("./Data/response.mp3")
    # os.system("start ./Data/response.mp3")
    pygame.mixer.init()
    pygame.mixer.music.load("./Data/response.mp3")
    pygame.mixer.music.play()

# Text To Speech

if __name__ == "__main__":
    # try:
    #     # Open the JSON file for reading
    #     with open("./Config/api.json", "r") as json_file:
    #         data = json.load(json_file)
    #         openai.api_key = data["openai"]
    # except FileNotFoundError:
    #     print(f"The file api.json does not exist.")
    try:
        # Open the JSON file for reading
        with open("./Config/api.json", "r") as json_file:
            data = json.load(json_file)
            os.environ["_BARD_API_KEY"] = data["bardkey"]
    except FileNotFoundError:
        print(f"The file api.json does not exist.")
# Speech To Text
    input_text = recognize_speech_from_microphone()
# ChatGPT
    if input_text:
        # generate_response_with_gpt(input_text)
        generate_response_with_bard(input_text)
    else:
        print("無法獲得輸入文字")


# %%
