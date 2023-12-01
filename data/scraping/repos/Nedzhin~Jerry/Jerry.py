import os
import openai
import time
import speech_recognition as sr
from fuzzywuzzy import fuzz
import pyttsx3
import datetime

opts = {
    "alias" :('jerry', 'джерри'),
    "tbr": ('скажи', 'расскажи', 'покажи', 'сколько', 'произнеси',),
    "cmds": {
        "ctime":('текущее время', 'который час', 'скажи время', 'время',),
        "SOS1":('вызови скорую', 'сокрую срочно', 'позвони в больницу', 'позвони в скорую'),
        "SOS2":('вызови полицию', 'полицию срочно', 'позвони в полицию', 'меня ограбили'),
        "SOS2":('пожар', 'поарных срочно', 'позвони пожарным'),
                                    }
    
    }

def speak(what):
    print(what)
    speak_engine.say(what)
    speak_engine.runAndWait()
    speak_engine.stop()




def callback(recognizer, audio):
    voice = recognizer.recognize_google(audio, language = "ru - RU ")
    print("Вы сказали:  " + voice.lower())

    try:

        print("[log] Распознано: " + voice)
        
        if voice.startswith(opts["alias"]):
            cmd = voice
            for x in opts['alias']:
                cmd = cmd.replace(x, "").strip()
            
            for x in opts['tbr']:
                cmd = cmd.replace(x, "").strip()
            
            cmd = recognize_cmd(cmd)
            print(cmd)
            execute_cmd(cmd['cmds'])
        return True
    except sr.UnknownValueError:

        print("[log] Не распознанно!" )
    except sr.RequestError as e:
        print("[log] Ошибка!")

def callbackPrint(input):
    cmd = input
    for x in opts['alias']:
        cmd = cmd.replace(x, "")
            
    for x in opts['tbr']:
        cmd = cmd.replace(x, "")
    cmd = recognize_cmd(cmd)
    if cmd['percent'] >60:
        execute_cmd(cmd['cmd'])
        return True
    else: return False
            
 
def recognize_cmd(cmd):
    RC = {'cmd': '', 'percent': 0}
    for c,v in opts['cmds'].items():
 
        for x in v:
            vrt = fuzz.ratio(cmd, x)
            if vrt > RC['percent']:
                RC['cmd'] = c
                RC['percent'] = vrt

    
    return RC

def execute_cmd(cmd):
    if cmd == 'ctime':
        # сказать текущее время
        now = datetime.datetime.now()
        speak("Сейчас " + str(now.hour) + ":" + str(now.minute))
    if cmd == 'SOS1':
        # сказать текущее время
        speak("Звоню в скорую!")
    


r = sr.Recognizer()

m = sr.Microphone(device_index = 1)

speak_engine = pyttsx3.init()

speak("Выберите Ввод с клавиатуры или Голосовое управлегие? Нажмите 1 Ввод с клавиатуры, Нажмите 2 Голосовое управлегие?")#
upravlenie = input()
if upravlenie == "1":
    userInput=''
    while userInput != 'выход':
        print('Чем я могу помочь?')
        userInput = input()
        if (callbackPrint(userInput) == False):
            openai.api_key = "YOUR_OPENAI_API_KEY"
            chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content":userInput}])
            speak(chat_completion["choices"][0]['message']['content'])
        
elif upravlenie == "2":
    query = ''
    while query != 'выход':
        with sr.Microphone(device_index=1) as source:
            r.adjust_for_ambient_noise(source)
            speak('чем я могу помочь?')
            audio = r.listen(source)
        query = r.recognize_google(audio, language="ru-RU")

        openai.api_key = "YOUR_OPENAI_API_KEY"
        print("Вы сказали:  " + query.lower())
        chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content":query.lower()}])

        speak(chat_completion["choices"][0]['message']['content'])

        en_voice='HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0'
        ru_voice = 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_RU-RU_IRINA_11.0 '

        voices = speak_engine.getProperty('voices')
        speak_engine.setProperty('voice', ru_voice) 

        stop_listening = r.listen_in_background(m, callback)
while True: time.sleep(0.1)


