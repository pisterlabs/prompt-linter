import speech_recognition as sr
from gtts import gTTS
import os
import openai

openai.api_key = "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"  # Replace with your actual OpenAI API key


def listen_for_speech():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    print("Listening...")
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print("You said: " + text)
        response = respond_to_speech(text)
        say_response(response)
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))


def respond_to_speech(text):
    print("Assistant: " + text)

    # Use OpenAI GPT-3 to generate a response based on the recognized speech
    response = generate_gpt3_response(text)

    return response


def generate_gpt3_response(input_text):
    # Use the OpenAI GPT-3 API to generate a response based on the input text
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=input_text,
        max_tokens=50,  # Adjust max_tokens as needed
        n=1,
        stop=None,
        temperature=0.7,  # Adjust temperature for creativity
    ).choices[0].text

    return response


def say_response(response):
    tts = gTTS(response)
    tts.save("output.mp3")
    os.system("mpg123 output.mp3")  # Use a command-line player to play the audio.


if __name__ == "__main__":
    while True:
        listen_for_speech()
