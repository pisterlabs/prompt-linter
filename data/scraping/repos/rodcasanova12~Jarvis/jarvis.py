# Python program to translate
# speech to text and text to speech

import openai
import speech_recognition as sr
import pyttsx3
import time

import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")

openai.api_key = OPENAI_KEY

# Function to convert text to Speech


def SpeakText(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()


# Initialize the recognizer
r = sr.Recognizer()


def record_text():
    # Loop in case of errors
    while (1):
        try:
            with sr.Microphone() as source2:
                r.adjust_for_ambient_noise(source2, duration=0.2)

                print("Please say something")

                audio2 = r.listen(source2)

                MyText = r.recognize_google(audio2)

                return MyText

        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

        except sr.UnknownValueError:
            print("unknown error occured")


MAX_TOKENS = 200


def send_to_ChatGPT(messages, model="gpt-3.5-turbo"):
    # Flatten the messages to check total tokens
    total_tokens = sum(len(message["content"]) for message in messages)

    # If the tokens exceed the maximum limit, you might want to drop some older messages
    while total_tokens > MAX_TOKENS:
        removed_message = messages.pop(0)  # remove the oldest message
        total_tokens -= len(removed_message["content"])

    try:
        response = openai.Completion.create(
            model=model,
            messages=messages,
            # Ensure you don't exceed the max tokens for the model
            max_tokens=MAX_TOKENS - total_tokens,
            n=1,
            stop=None,
            temperature=0.5,
        )
        response_message = response.choices[0].message['content']
        return response_message
    except openai.error.RateLimitError:
        print("Rate limit exceeded. Please wait for a while before making another request.")
        # Pause for 15 seconds as a precaution; adjust as needed
        time.sleep(15)
        return "Rate limit exceeded. Please wait."
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred: {e}"


messages = [{"role": "user", "content": "Please act like Jarvis from Iron man."}]
while (1):
    text = record_text()
    messages.append({"role": "user", "content": text})
    response = send_to_ChatGPT(messages)
    SpeakText(response)

    print(response)
