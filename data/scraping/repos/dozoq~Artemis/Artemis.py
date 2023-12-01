import os
import time
import openai
import pyaudio
import playsound
from gtts import gTTS
import speech_recognition as sr
import pyttsx3
import resources as r
import ignored


class Artemis:
    AIName = "Artemis"
    voiceID = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_PL-PL_ZIRA_11.0"

    def __init__ ( self, openAPIEnvironmentKey: str, langShortCode: str = "en", microphoneIndex: int = 0 ):
        self.openAPIEnvironmentKey = openAPIEnvironmentKey
        self.langShortCode = langShortCode.lower()
        self.initialized = False
        self.recognizer = sr.Recognizer()
        self.microphoneIndex = microphoneIndex
        self.player = pyttsx3.init()
        self.player.setProperty("rate", 150)
        self.player.setProperty("volume", 1)
        self.player.setProperty("voice", Artemis.voiceID)
        #self.player.say(r.Resources["Welcome"])
        #self.player.runAndWait()

    def WaitForCommand ( self ):
        while True:
            with sr.Microphone( self.microphoneIndex ) as source:
                audio = self.recognizer.listen(source)
                try:
                    said = self.recognizer.recognize_google(audio, language="pl-PL")
                    print(said)
                    if Artemis.AIName in said:
                        said.replace(Artemis.AIName, "")
                        completition = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role":"user", "content": said}])
                        text = completition.choices[0].message.content
                        self.player.say( text )
                        self.player.runAndWait()
                    else:
                        self.player.say( "Nie usłyszałam komendy" )
                        self.player.runAndWait()
                except Exception as exc:
                    print(exc)
    def GetAudio ( self ):
        pass

    def Calibrate ( self ):
        self.player.say( r.Resources["Calibration"] )
        self.player.runAndWait()
        openai.api_key = ignored.APIKey
        self.player.say( r.Resources["CalibrationEnd"] )
        self.player.runAndWait()
        self.initialized = True

    def GetModelsList ( self ):
        self.CheckInitialization()
        return openai.Model.list()

    def CheckInitialization ( self ):
        if not self.initialized:
            raise Exception( "Initialize First" )
