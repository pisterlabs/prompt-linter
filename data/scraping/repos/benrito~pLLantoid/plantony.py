from dotenv import load_dotenv
import pyaudio
import wave
import time
import openai
import requests
import json
from playsound import playsound
import tempfile
import os
import sounddevice as sd
import numpy as np
import random
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import speech_recognition as sr
import subprocess
import threading

# Load environment variables from .env file
load_dotenv()

# Access environment variables
openai.api_key = os.environ.get("OPENAI")
eleven_labs_api_key = os.environ.get("ELEVEN")

# Load the sounds
beep_start = 'media/beep_start.mp3'
beep_stop = 'media/beep_stop.wav'
reflection = 'media/initiation.mp3'
cleanse = 'media/cleanse.mp3'
ambient_sounds = [
    'media/ambient1.mp3',
    'media/ambient2.mp3',
    'media/ambient3.mp3',
    'media/ambient4.mp3'
]
acknowledgement_sounds = [
    'media/acknowledgement1.mp3',
    'media/acknowledgement2.mp3',
    'media/acknowledgement3.mp3',
    'media/acknowledgement4.mp3'
]
radiation = random.choice(ambient_sounds)
acknowledgement = random.choice(acknowledgement_sounds)

# create the seed
timestamp = str(int(time.time()))
with open('seed_words.json', 'r') as file:
    word_categories = json.load(file)
selected_words = []
for category in word_categories:
    selected_words.append(random.choice(category['items']))
selected_words_string = ', '.join(selected_words)
print("My seed words are: " + selected_words_string)

# listen for input
time.sleep(1)  # wait for 1 second
playsound(reflection)
with sr.Microphone() as source:
    print("I'm listening...")
    audio = r.listen(source)
    # Save the audio data to a WAV file
    with open(f"recordings/{timestamp}_recording.wav", "wb") as f:
        f.write(audio.get_wav_data())

# Use the recognizer to convert speech to text, playing some atmospherics while we wait
try:
    def ambient_background(radiation, stop_event):
        while not stop_event.is_set():
            playsound(radiation)

    # Create and start the thread
    stop_event = threading.Event()
    sound_thread = threading.Thread(target=ambient_background, args=(radiation, stop_event))
    sound_thread.daemon = True
    sound_thread.start()

    # Recognize the speech input using Google Speech Recognition
    playsound(acknowledgement)
    text = r.recognize_google(audio)
    print("I heard: " + text)

    # Prepare to pass the transcript to the prompt
    generated_transcript = text
    filename = f"transcripts/{timestamp}_transcript.txt"

    # Save the generated transcript to a file with epoch timestamp
    with open(f"transcripts/{timestamp}_transcript.txt", "w") as f:
        f.write(generated_transcript)
        print(f"Transcript saved to transcripts/{timestamp}_transcript.txt")

except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")

except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))

# Prepare for magic

# The GPT-3.5 model ID you want to use
model_id = "text-davinci-003"

# The maximum number of tokens to generate in the response
max_tokens = 1024

# Construct the prompt with the embedded transcript
prompt = f"You are Plant-Tony, an enlightened being from the future. Answer the following qestion in the form of a thoughtful poem:\n\n{generated_transcript}\n\nInclude the following words in your poem: {selected_words_string} :"

# Generate the response from the GPT-3.5 model
response = openai.Completion.create(
    engine=model_id,
    prompt=prompt,
    max_tokens=max_tokens
)

# Save the response to a local file with an epoch timestamp
filename = f"responses/{int(time.time())}_response.txt"
with open(filename, "w") as f:
    f.write(response.choices[0].text)
    print(f"Output saved to responses/{timestamp}_response.txt")
sermon_text = response.choices[0].text

# Now let's read it...

voice = "https://api.elevenlabs.io/v1/text-to-speech/o7lPjDgzlF8ZloHzVPeK"

# Read the text from the input file
with open(filename, "r") as f:
    text = f.read()

# Choose a random URL string from the list
url = (voice)

headers = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": eleven_labs_api_key
}

    # Request TTS from remote API
response = requests.post(url, json={"text": text, "voice_settings": {"stability": 0, "similarity_boost": 0}}, headers=headers)
print(response.status_code)
if response.status_code == 200:
    # Save remote TTS output to a local audio file with an epoch timestamp
    filename = f"sermons/{timestamp}_sermon.mp3"
    with open(filename, "wb") as f:
        f.write(response.content)
        print(f"Sermon saved to sermons/{timestamp}_sermon.mp3")
        
    # Play the audio file and cleanse
    print(sermon_text)
    playsound(filename)
    time.sleep(2)  # wait for 2 seconds
    playsound(cleanse)

# End the script
exit()
