import speech_recognition as sr
import pyttsx3 as tts
from openai_agent import OpenAIAgent
import time
from pygame import mixer

class SpeechProcessing:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.tts_engine = tts.init()
        self.openai_agent = OpenAIAgent()
        self.sound_file = "listen_sound.mp3"

        mixer.init()

        self.tts_engine.setProperty("voice", "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0")
        self.tts_engine.setProperty("rate", 178)

    def play_sound(self):
        mixer.music.load(self.sound_file)
        mixer.music.play()

    def listen_for_wakeword(self):
        wakeword = "hello my friend"
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Waiting for wake word...")
            try:
                audio = self.recognizer.listen(source, timeout=5)
                text = self.recognizer.recognize_google(audio)
                if text.lower() == wakeword:
                    print("Wake word detected.")
                    return True
                    
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                print("Couldn't request results from the Google Speech Recognition service")
            except Exception as e:
                print(f"There was an error: {e}")
            
            return False
                    

    def listen(self, timeout=5):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            self.play_sound()
            print("Listening...")
            audio = None
            try:
                audio = self.recognizer.listen(source, timeout)
            except sr.WaitTimeoutError:
                print("Listening timed out while waiting for phrase to start")
                return ""
                
            text = ""

            try:
                print("Recognizing...")
                text = self.recognizer.recognize_google(audio)
                print(f"User said: {text}")
            except sr.UnknownValueError:
                print("Google Speech could not recognize audio")
            except sr.RequestError:
                print("Couldn't request results from the Google Speech Recognition service")
            except Exception:
                print("There was an error")
            
            return text

    def speak(self, text, rephrase=True):
        self.queue(text, rephrase)
        self.runAndWait()

    def queue(self, text, rephrase=True):
        if rephrase:
            self.tts_engine.say(self.openai_agent.rephrase(text))
        else:
            self.tts_engine.say(text)

    def runAndWait(self):
        self.tts_engine.runAndWait()