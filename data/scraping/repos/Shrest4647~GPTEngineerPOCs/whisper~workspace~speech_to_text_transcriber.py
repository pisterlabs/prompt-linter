import os
import time
from datetime import datetime
from dotenv import load_dotenv
import speech_recognition as sr
import openai
from chatbot_api import ChatbotAPI
from text_to_speech import TextToSpeech

load_dotenv()

class SpeechToTextTranscriber:
    def __init__(self, codeword_start: str, codeword_pause: str, codeword_end: str, idle_timeout: int, language: str, accent: str):
        self.codeword_start = codeword_start
        self.codeword_pause = codeword_pause
        self.codeword_end = codeword_end
        self.idle_timeout = idle_timeout
        self.language = language
        self.accent = accent
        self.transcribing = False
        self.paused = True

        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.chatbot_api = ChatbotAPI(os.getenv("OPENAI_API_KEY"))
        self.text_to_speech = TextToSpeech()

    def start_transcribing(self):
        self.transcribing = True
        self.listen_for_codewords()

    def stop_transcribing(self):
        self.transcribing = False

    def listen_for_codewords(self):
        while self.transcribing:
            audio = MicrophoneListener().listen()
            transcription = self.transcribe(audio)
            if self.codeword_pause.lower() in transcription.lower():
                self.paused = True
            elif len(transcription) > 0 and (self.codeword_start.lower() in transcription.lower() or not self.paused):
                self.paused = False
                self.save_transcription(transcription)
                self.ask_question(transcription)
            else:
                time.sleep(self.idle_timeout)
            if self.codeword_end.lower() in transcription.lower():
                self.stop_transcribing()

    def transcribe(self, audio: bytes) -> str:
        try:     
            r = sr.Recognizer()
            my_string=r.recognize_google(audio, language=self.language, show_all=False)
            print(my_string)
            return my_string
          
        except Exception as e:
            print(e)
            return ""

    def save_transcription(self, transcription: str):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"transcription_{timestamp}.txt"
        with open(filename, "w") as file:
            file.write(transcription)
    
    def ask_question(self, transcription: str):
        self.transcribing = False
        try:
            print("Asking question...")
            question = transcription.lower().split(self.codeword_start.lower())[1].strip() if self.codeword_start.lower() in transcription.lower() else transcription.lower().strip()
            print("Question:", question)
            answer = self.chatbot_api.send_message(question)
            print("Answer:", answer)
            self.text_to_speech.speak(answer)
        except Exception as e:
            print(e)
        finally:
            self.transcribing = True



class MicrophoneListener:
    def __init__(self):
        self.microphone = sr.Microphone(device_index=1)

    def listen(self) -> bytes:
        recognizer = sr.Recognizer()

        with self.microphone as source:
            print("Recording...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        return audio
