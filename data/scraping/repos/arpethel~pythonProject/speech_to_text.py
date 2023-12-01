# speech_recognition.py
import os
import openai
import speech_recognition as sr

class SpeechToText:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.audio = None
        self.transcript = ""
        # Set OpenAI key
        openai.api_key = os.getenv('OPENAI_KEY')

    def start_recording(self):
        with sr.Microphone() as source:
            print("Recording...")
            self.audio = self.recognizer.listen(source)

    def stop_recording_and_transcribe(self):
        try:
            self.transcript = self.recognizer.recognize_google(self.audio)
            print("Transcript Generated.")
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand your audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {0}".format(e))

    def generate_response(self):
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=self.transcript,
            max_tokens=150
        )
        return response.choices[0].text.strip()
