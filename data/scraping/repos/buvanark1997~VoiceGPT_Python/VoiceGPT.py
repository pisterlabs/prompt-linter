import os
import time
import openai
import pyttsx3
import speech_recognition as sr
import pyaudio
openai.api_key="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"
engine = pyttsx3.init()
def transcribe_audio_to_text(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        print('skipping known error')

def generate_response(prompt):
    print(f"Came inside generate_response {prompt}")
    response = openai.Completion.create(
        engine="text-davinci-003",
        max_tokens=4000,
        prompt = prompt,
        n=1,
        stop=None,
        temperature=0.5,
    )
    print(response)

    return response["choices"][0]["text"]


def speak_text(text):
    engine.say(text)
    engine.runAndWait()

def main():
    while True:
        print("say 'Genius' to start recording your question...")
        with (sr.Microphone() as source):
            recognizer = sr.Recognizer()
            audio = recognizer.listen(source)
            try:
                transcription = recognizer.recognize_google(audio)
                print(transcription)
                if transcription.lower() == "genius":
                    print("got inside genius")
                    filename = "input.wav"
                    print("say your question...")
                    with sr.Microphone() as source:
                        print("recording start")
                        recognizer = sr.Recognizer()
                        source.pause_threshold = 1
                        audio = recognizer.listen(source, timeout=2)
                        print(audio)
                        with open(filename, "wb") as f:
                            f.write(audio.get_wav_data())
                            # transcribe audio to text
                        text = transcribe_audio_to_text(filename)
                        print(text)

                        if text:
                            print(f"you said:{text}")
                            # generate response using GPT-3
                            response = generate_response(text)
                            print(f"GPT-# says:{response}")

                            # Read response using text-to-speech
                            speak_text(response)
            except Exception as e:
                print("An error occured:{}",format(e))
main()