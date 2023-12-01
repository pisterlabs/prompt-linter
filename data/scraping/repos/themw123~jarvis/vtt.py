import os
import speech_recognition as sr
from speech_recognition import Recognizer
from speech_recognition import AudioData
import tempfile
import openai

class Vtt:
    def __init__(self, recognition_conf, chatgpt_conf):
        self.recognition_conf = recognition_conf
        self.chatgpt_conf = chatgpt_conf
    
    
    def vtt_wrapper(self, audio: AudioData):
        print("- interpret...")
        if self.recognition_conf["active"] == "whisper":
            text = self.__vtt_whisper(audio)
        elif self.recognition_conf["active"] == "google":
            r = Recognizer()
            text = self.__vtt_google(r, audio)
        else:
            raise Exception(self.recognition_conf["active"] + ": This vtt api type does not exist") 
        
        text = text.replace("\n", "")
        return text

    def __vtt_whisper(self, audio: AudioData):

        text = None

        try:
            # Erstelle eine temporäre Datei und schreibe den Inhalt des Audio-Streams hinein
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_file.write(audio.get_wav_data())
        except Exception as e:
            raise Exception("Temporäre Datei konnte nicht erstellt werden")

        try:
            with open(temp_file.name, "rb") as audio_file:
                openai.api_key = self.chatgpt_conf["official"]["api_key"]
                text = openai.Audio.transcribe(
                    file = audio_file,
                    model = "whisper-1",
                    response_format="text",
                    language=self.recognition_conf["whisper"]["language"]
                ) 
        except IOError as file_error:
            # Exception for file reading errors (e.g., file not found, permissions issue)
            raise Exception("Fehler beim lesen der Temp File:", file_error)
        except Exception as api_error:
            # Exception for API request errors (e.g., network issues, invalid API response)
            raise Exception("Fehler bei der Whisper API:", api_error)
        finally:
            temp_file.close()
            os.remove(temp_file.name)    
                
        return text
    
    
    def __vtt_google(self, r: Recognizer, audio: AudioData):
        text = ""
        try:
            #nur google ist ohne api key.
            text = r.recognize_google(audio, language=self.recognition_conf["google"]["language"])
        except sr.UnknownValueError:
            print("- Sprache konnte nicht erkannt werden.\n")
        except sr.RequestError:
            print("- Fehler beim Abrufen der Spracherkennung.\n")
        return text