#----------------------------------------------------
# JARVIS voice assistant
# copyright fab.gan@hotmail.de
#----------------------------------------------------

import time
import speech_recognition as sr
import os
#import openai
import pyttsx3
import json
from magichome import *
from flux_led import BulbScanner, LedTimer, WifiLedBulb

engine = pyttsx3.init()

voices = engine.getProperty('voices')
engine.setProperty('rate', 150) 
engine.setProperty('voice', voices[0].id)

recognizer = sr.Recognizer()
microphone = sr.Microphone()
threshold = 120 # 2 minuten

ip_bulk = '192.168.0.240'

with microphone as source:
  while True:
    recognizer.adjust_for_ambient_noise(source)
    print("JARVIS bereit")

    engine.say("j.a.r.v.i.s bereit")
    engine.runAndWait()
  
    called = False

    audio = recognizer.listen(source, phrase_time_limit=5)
    try:
      transcription = recognizer.recognize_google(audio,language = "de-DE", show_all=True)
      print(transcription)
      if transcription != []:
        for trans1 in transcription["alternative"]:
          if "jarvis" in trans1["transcript"].lower():
            engine.say("ja")
            engine.runAndWait()

            correct = False
            question_time = time.time()

            while not correct and (time.time() - question_time < threshold):
              try: 
                print("höre zu...")
                audio = recognizer.record(source, duration=5)
                transcription = recognizer.recognize_google(audio, language = "de-DE", show_all=True)
                if transcription != []:
                  correct = False
                  for trans in transcription["alternative"]:
                    if "test" in trans["transcript"].lower():
                      engine.say("BITCH")
                      engine.runAndWait()
                      correct = True
                      break
                    elif "wer bin ich" in trans["transcript"].lower():
                      engine.say("Eine sehr nette Person die auch noch sehr gut aussieht. Einfach ein geiler Hase")
                      engine.runAndWait()
                      correct = True
                      break
                    elif "licht an" in trans["transcript"].lower():
                      engine.say("logo meister")
                      engine.runAndWait()

                      bulb = WifiLedBulb(ip_bulk)
                      bulb.refreshState()
                      bulb.setRgb(220,220,220)

                      correct = True
                      break
                    elif "licht aus" in trans["transcript"].lower():
                      engine.say("logo meister")
                      engine.runAndWait()

                      bulb = WifiLedBulb(ip_bulk)
                      bulb.turnOff()

                      correct = True
                      break
                  if not correct:
                    engine.say("no comprendo")
                    engine.runAndWait()
              except sr.UnknownValueError:
                print("Unable to transcribe audio")
                engine.say("Ich hatte ein Problem mit dem Audio, bitte wiederholen")
                engine.runAndWait()
    except sr.UnknownValueError:
      print("UnknownValueError")
