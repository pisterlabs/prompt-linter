import subprocess
# import speech library
import pyttsx3
# import speech recognition libraries
import speech_recognition as sr
import wave
import tempfile
import struct
import numpy as np
# import pvrecorder and porcupine libraries
from pvrecorder import PvRecorder
import pvporcupine
# import openai library
import openai
# import env libraries
import json
import os
from dotenv import load_dotenv
# import time library
import time

# load .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
pico_api_key = os.getenv("PICO_API_KEY")
wake_word = os.getenv("WAKE_WORD")

tempfile_name = os.path.join(tempfile.gettempdir(), "command.wav")
silence_threshold = 1000  # Adjust this threshold as needed

# set up variable to check if first run
first_run = True

# set up command lookup array
command_lookup = {
    "conversation": ["echo", ""],
    "play discover weekly": ["osascript", "-e", 'tell application "Spotify" to play track "spotify:playlist:37i9dQZEVXcMaWCjUILjal"'],
    "exit": ["exit"],
}
commands_string = str(list(command_lookup.keys()))  # Convert array to string

# set up speech recognition
porcupine = pvporcupine.create(
    access_key=pico_api_key,
    keyword_paths=["wake_word/wake_word.ppn"],
)
recognizer = sr.Recognizer()

# set up text to speech
engine = pyttsx3.init()
engine.setProperty('rate', 190)

def get_silence_threshold():
    command_recorder = PvRecorder(device_index=-1, frame_length=512)
    audio = []
    try:
        command_recorder.start()

        start_time = time.time()
        volume = 0  # total volume of frames

        while time.time() - start_time < 2:
            frame = command_recorder.read()
            audio.extend(frame)
        command_recorder.stop()

        volume = np.abs(frame).mean()
        return volume
    finally:
        command_recorder.delete()

def record_command():
    command_recorder = PvRecorder(device_index=-1, frame_length=512)
    audio = []

    try:
        command_recorder.start()

        start_time = time.time()

        while time.time() - start_time < 5:
            frame = command_recorder.read()
            audio.extend(frame)

            if len(frame) >= 30:
                volume = np.abs(frame[-31:]).mean()
                if volume * 1.2 <= silence_threshold:
                    print("Silence detected. Stopping recording.")
                    break

        command_recorder.stop()
        with wave.open(tempfile_name, 'w') as f:
            f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
            f.writeframes(struct.pack("h" * len(audio), *audio))
    finally:
        command_recorder.delete()

    
def say(text):
    print(text)
    engine.say(text)
    engine.runAndWait()

def execute_command(command):
    subprocess.call(command)

def handle_response(response):
    try:
        assistant = json.loads(response)
    except json.decoder.JSONDecodeError:
        say("Error decoding JSON response")
        return
    
    if assistant['message'] is not None:
        say(assistant['message'])
    elif assistant['command'] is not None:
        if assistant['command'] == ["exit"]:
            print("Exiting...")
            exit()
        else:
            execute_command(assistant['command'])
    else:
        print("Command not recognized.")
        return

def get_gpt(command):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are Jarvis, a dry, snarky, ai assistant. Based on the below input, select the best command to execute from the following array. Return only the command that should be executed from the array and a message as a JSON object. For conversation give a reply via the message key. Only give the json object. Commands:" + commands_string},
            {"role": "user", "content": "Input:" + command}
        ]
    )

    if response["choices"][0]["message"]["role"] == "assistant":
        gpt_response = response["choices"][0]["message"]["content"]
        print("GPT-3:", gpt_response)
        handle_response(gpt_response)
    else:
        say("Sorry, something went wrong with my brain.")

say("Initializing...")

# Initialize speech recognizer
r = sr.Recognizer()

recorder = PvRecorder(
        frame_length=porcupine.frame_length,
        device_index=-1)
recorder.start()

say("Calibrating Noise Level. Please be quiet for 2 seconds.")

silence_threshold = get_silence_threshold()

say("Initialized. Silence threshold: " + str(silence_threshold) + ".")

try:
    # Infinite loop for continuous listening
    while True:
        recorder.start()
        if first_run:
            say("Listening for wake word...")
            first_run = False
        
        # Use pvrecorder to record audio until wake word is detected
        while True:
            pcm = recorder.read()
            keyword_index = porcupine.process(pcm)
            if keyword_index >= 0:
                print("Wake word detected!")
                recorder.stop()
                break
        # Use speech recognition after wake word is detected
        print("Listening for command...")
        say("Yes?")
        record_command()
        with sr.AudioFile(tempfile_name) as source:
            audio = r.record(source)

        try:
            # Perform speech recognition
            command = r.recognize_google(audio)
            print(f"Command: {command}")
            say("Processing...")
            get_gpt(command)
        except sr.UnknownValueError:
            say("Sorry, I didn't understand that.")
        except sr.RequestError as e:
            print(f"Error: {str(e)}")
            say("Error. Please try again.")
except KeyboardInterrupt:
    print("Stopping...")
    recorder.stop()
    recorder.delete()
    porcupine.delete()
    exit()
