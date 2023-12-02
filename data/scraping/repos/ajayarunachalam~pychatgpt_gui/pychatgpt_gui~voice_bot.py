#! /usr/bin/env python
"""
@author: Ajay Arunachalam
Created on: 30/04/2023
Goal: Voice BOT CHATGPT 
Version: 0.0.1
"""
import openai
import os, json
import sys
import playsound
import speech_recognition as sr

from typing import Text
from gtts import gTTS


# Initialize the recognizer
r = sr.Recognizer()

# Set up the OpenAI API client
# Set up the OpenAI API client

with open('../api_key.json', 'r') as f:
    openai_key = json.load(f)
    print(openai_key)
    
    if 'api_key' in openai_key:
        openai.api_key = openai_key['api_key']

def speak_chatgpt_text(text: str):
    # Initialize gTTS engine
    lang_accent = 'com.en'
    filename = "tmp.mp3"
    tts = gTTS(text, tld=lang_accent)
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)


def ask_chatgpt(prompt: str) -> Text:
    chat_gpt3_model_engine = "text-davinci-003"
    results = []
    # Generate a streamed response
    for resp in openai.Completion.create(engine=chat_gpt3_model_engine, prompt=prompt, max_tokens=512, n=1, stop=None, temperature=0.5, stream=True, ):
        text = resp.choices[0].text
        results.append(text)
        sys.stdout.write(text)
        sys.stdout.flush()

    return "".join(results)


def run_voice_bot():
    while True:
        # Exception handling to handle exceptions at runtime if
        # no user prompt given
        try:
            # use the microphone as source for input.
            with sr.Microphone() as source2:
                print("Microphone is open now say your prompt...")
                # wait for a second to let the recognizer
                # adjust the energy threshold cbased on
                # the surrounding noise level
                r.adjust_for_ambient_noise(source2, duration=0.2)

                # listens for the user's input
                audio2 = r.listen(source2)

                # Using google to recognize audio
                my_prompt = r.recognize_google(audio2)
                my_prompt = my_prompt.lower()

                print("Did you say :", my_prompt)
                prompt_resp_text = ask_chatgpt(my_prompt)
                speak_chatgpt_text(prompt_resp_text)

        except Exception as e:
            print(e)
            print("Could not request results; {0}".format(e))


if __name__ == '__main__':

    configured_microphones = sr.Microphone.list_microphone_names()
    if configured_microphones:
        for index, name in enumerate(configured_microphones):
            print("Microphone with name \"{1}\" found for microphone(device_index{0})".format(index, name))
        run_voice_bot()
    else:
        print("No configured Microphones found")