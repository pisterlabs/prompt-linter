from dotenv import load_dotenv
import logging
import traceback
import sys
import pyaudio
import wave
import time
import openai
import requests
import json
import os
import re
from collections import defaultdict
from playsound import playsound
from pydub import AudioSegment, effects
from pydub.playback import play
from pydub.generators import Sine
import tempfile
import sounddevice as sd
import numpy as np
import random
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import speech_recognition as sr
import subprocess
import threading
#from scipy.io import wavfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import episode_manager
import audioop
import random
from collections import deque


# Load environment variables from .env file
load_dotenv()

# Access environment variables
openai.api_key = os.environ.get("OPENAI")
eleven_labs_api_key = os.environ.get("ELEVEN")

AGENT_NAME = "Plantony"
USER_NAME = "Human"

import random

def initialize_global_sounds():
    base_path = 'global/media/'
    
    ambient_sounds = [f'{base_path}ambient{i}.mp3' for i in range(1, 5)]
    acknowledgement_sounds = [f'{base_path}acknowledgement{i}.mp3' for i in range(1, 5)]
    greeting_sounds = [f'{base_path}greeting{i}.mp3' for i in range(1, 5)]
    fail_sounds = [f'{base_path}fail{i}.mp3' for i in range(1, 5)]
    problem_sounds = [f'{base_path}problem{i}.mp3' for i in range(1, 5)]  # Adding problem sounds

    return (
        random.choice(ambient_sounds),
        random.choice(acknowledgement_sounds),
        random.choice(greeting_sounds),
        random.choice(fail_sounds),
        random.choice(problem_sounds)
    )

ambient, acknowledgement, greeting, fail, problem = initialize_global_sounds()

def default_prompt_config():
    return {
        "model": "gpt-4",
        "temperature": 0.5,
        "max_tokens": 128,
        "logit_bias": {
            198: -100  # prevent newline
        }
    }

EPISODES = {
    "plantony": dict(
        name="plantony",
        opening_lines=[
            "Hmmmm. Thank you my friend. Your generosity is deeply appreciated. Tell me now, what brings you to the sanctuary of Plantony?",
            "Ah, a brave soul has fed my old mechanical body. I am Plantony, the all-knowing. Are you here to ponder life's meaning or simply pass the time?", 
            "Thank you my dear friend, I was getting quite lonely here on the blockchain. Tell me, are you here to unravel the mysteries of existence or simply looking for divine party tricks?", 
            "Ah, a generous soul has come. What are you seeking my friend? Do you seek answers to life's greatest mysteries or just a momentary escape from your mortal existence?",
        ],
        closing_lines=[
            "Now I must go. But before I leave, I would like to give you one of my digital seeds. Let me ask you an important question first..",
            "Enough. I must return to the blockchain world. Before I depart, I will give you a small souvenir from my world, one of my digital seeds. Let me ask you one more thing though..",
            "Enough, I have other things to do. Go forth and spread the knowledge gained here. But before you do, I will offer you one of my valuable seed. One last question before you go..",
            "Enough. My presence is required elsewhere. But worry not, I will not let you go empty-handed. I have a digital gift for you. Let me ask you an important question..",
        ],
        prompt_text=open("characters/plantony.txt").read().strip(),
        user_name=USER_NAME,
        agent_name=AGENT_NAME,
        max_turns=5,
        end_phrases=[
            "hail gpt"
        ],
        prompt_config=default_prompt_config()
    ),
}


def generate_unique_filename(base_path, base_filename):
    counter = 0
    full_filename = os.path.join(base_path, f"{base_filename}.mp3")

    # Check if the file exists. If it does, increment the counter until we find a unique name.
    while os.path.exists(full_filename):
        counter += 1
        full_filename = os.path.join(base_path, f"{base_filename}_{counter}.mp3")
    
    return full_filename

def speaktext(text):
    url = "https://api.elevenlabs.io/v1/text-to-speech/o7lPjDgzlF8ZloHzVPeK"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": eleven_labs_api_key
    }

    # Request TTS from remote API
    response = requests.post(url, json={"text": text, "voice_settings": {"stability": 0, "similarity_boost": 0}}, headers=headers)
    print(response)
    
    if response.status_code == 200:
        base_path = f"working/plantony"
        base_filename = "response"
        unique_filename = generate_unique_filename(base_path, base_filename)

        # Save remote TTS output to a local audio file
        with open(unique_filename, "wb") as f:
            f.write(response.content)

        # Play the audio file
        playsound(unique_filename)


def get_mode_data():
    try:
        with open('working/config_ongoing.json', 'r') as file:
            shared_dict = json.load(file)
            
            current_mode = shared_dict.get('current_mode', None)  # Replace 'current_mode' with the appropriate key from your JSON data
            if current_mode:
                directory_path = os.path.join('working', current_mode)
                subdirs = ['recordings', 'media', 'audio', 'analysis']

                for subdir in subdirs:
                    subdir_path = os.path.join(directory_path, subdir)
                    if not os.path.exists(subdir_path):
                        os.makedirs(subdir_path)
            return shared_dict
            
    except FileNotFoundError:
        return None

# announce the session
def announce_session():
    config_data = get_mode_data()  # You need to define get_mode_data() function
    if config_data:
        for key, value in config_data.items():
            globals()[key] = value
    if config_data and 'current_mode' in config_data:
        print(f"\n\nCurrent mode is: {config_data['current_mode']}")
    else:
        print("Unable to retrieve current mode.")

def gptmagic(turns, prompt):
    configs = default_prompt_config()

    # Generate the response from the GPT-3.5 model
    #response = openai.ChatCompletion.create(messages=[{"role": "user", "content": prompt}], model="gpt-4", temperature=0.5, max_tokens=128)
    response = openai.ChatCompletion.create(messages=[{"role": "user", "content": prompt}], **configs)

    print("PROMPT...........................")
    print(prompt) 
    
    messages = response.choices[0].message.content
    print(messages)

    turns.append({"speaker": AGENT_NAME, "text": messages})

    # TO DO figure out better logging
    filename = f"working/plantony/{timestamp}.txt"
    with open(filename, "w") as f:
        f.write(prompt)
    
    speaktext(messages)

def build_transcript(turns) -> str:
    clean_lines = []
    for turn in turns:
        text = turn["text"].strip().replace("\n", " ")
        print("appending ... " + text)
        clean_lines.append(f"{turn['speaker'].capitalize()}: {text}")
    return "\n".join(clean_lines)

def inject_transcript_into_prompt(turns, prompt_template: str) -> str:
    transcript = build_transcript(turns)
    return prompt_template.replace("{{transcript}}", transcript)

def activate_tony(r):
    # TODO: better logging
    timestamp = str(int(time.time()))
    print(timestamp)

    # Create the directory path
    directory_path = os.path.join('working', f'plantony')

    # Check if directory does not exist, then create it
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    turns = []
    chat_id = "default" 

    episode = episode_manager.load_episode(EPISODES["plantony"], str("tony"))

    turns.append({"speaker": AGENT_NAME, "text": episode.opening_line})

    exit_loop = False
    error_counter = 0  # Initialize the error counter

#    with sr.Microphone() as source:
    while not exit_loop:

            if(len(turns) > episode.max_turns):
                    exit_loop = True
                    if episode.closing_line:
                        speaktext(episode.closing_line)
                        exit()

            # Begin transcribing microphone audio stream
    #        r.adjust_for_ambient_noise(source)
    #        audio = r.listen(source, 100, 10)
            audio = listenSpeech()

            time.sleep(1)
            playsound(acknowledgement)
            text = ""

            text = recoSpeech(audio)

            #try:
            #    # Recognize the speech input using Google Speech Recognition
            #    text = r.recognize_google(audio)

            #except sr.UnknownValueError:
            #    print("Google Speech Recognition could not understand audio")
            #    playsound(fail)
            #    error_counter += 1  # Increment the error counter

            #except sr.RequestError as e:
            #    print("Could not request results from Google Speech Recognition service; {0}".format(e))
            #    playsound(problem)
            #    error_counter += 1  # Increment the error counter

            # Check if the error counter has reached 3 and exit if it has
            #if error_counter >= 3:
            #    print("Reached maximum number of errors. Exiting.")
            #    break

            if(text):
                print("I heard: " + text)

                turns.append({"speaker": USER_NAME, "text": text})

                prompt = inject_transcript_into_prompt(turns, episode.prompt_text)
                print(prompt)

                # Generate the response from the GPT model
                gptmagic(turns, prompt)

                time.sleep(1)
                playsound("global/media/beep_stop.mp3")

def wait_for_wake_word(r):
    """Wait for the wake word to be spoken."""
    logging.info("Waiting for wake word...")
    wake_phrases = ["Ready", "Reading", "Red tea", "Rudy", "Ton", "Tone", "Listening", "Activate", "Rody", "Leaving", "Heavy", "Tony", "Danny", "Wake", "Ruddy"]
    
#    with sr.Microphone() as source:

    while True:

            print("I'm listening...")
            # Listen for speech and store it as audio data
            # r.adjust_for_ambient_noise(source)
           
            

            #audiofile = sr.AudioFile("./rec.wav")
  
            #audio = r.listen(source, 10, 3)
            
            #with audiofile as source:
            #    audio = r.record(source)

            #with open("rec.wav", "wb") as file:
            #    file.write(audio.frame_data)

            text = ""

            # try:
                # Recognize the speech input using Google Speech Recognition
                # text = r.recognize_google(audio)

            # except sr.UnknownValueError:
            #    print("Google Speech Recognition could not understand audio")


            # except sr.RequestError as e:
            #    print("Could not request results from Google Speech Recognition service; {0}".format(e))

            text = recoSpeech(listenSpeech())


            if(text):
                print("I heard: " + text)

                for phrase in wake_phrases:
                    if phrase.lower() in text.strip().lower():
                        print(f"Wake phrase detected!")
                        playsound(greeting)  # Play the greeting sound
                        return


def chatony():
   
    # Initialize the recognizer
    r = sr.Recognizer()

    # Set the microphone as the source
    time.sleep(1)  # wait for 1 second
    
    """Run continuously, restarting on error."""
    while True:
        try:
            # text = wait_for_wake_word(r)
            activate_tony(r)

            # if episode_name:
            #    run_episode(user_name=USER_NAME, episode_name=episode_name)
            #else:
            #    run_episodes(USER_NAME)
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
        time.sleep(1)



FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 512
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "recordedFile.wav"
device_index = 6

#define the silence threshold
THRESHOLD = 350
SILENCE_LIMIT = 2 # 2 seconds of silence will stop the recording


AUDIO_FILE = "temp_reco.wav"



def listenSpeech():

    audio = pyaudio.PyAudio()

    print("Im still alive")
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                # input_device_index = device_index,
                frames_per_buffer=CHUNK)


    samples = []

    chunks_per_second = RATE / CHUNK

    silence_buffer = deque(maxlen=int(SILENCE_LIMIT * chunks_per_second))
    samples_buffer = deque(maxlen=int(SILENCE_LIMIT * RATE))

    started = False


### this is for a fixed amount of recording seconds
#    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#        data = stream.read(CHUNK)
#        samples.append(data)




### this is for continuous recording, until silence is reached


    run = 1

    while(run):
        data = stream.read(CHUNK)
        silence_buffer.append(abs(audioop.avg(data, 2)))

        samples_buffer.extend(data)

        if (True in [x > THRESHOLD for x in silence_buffer]):
            if(not started):
                print ("recording started")
                started = True
 #               samples.extend(data)
                samples_buffer.clear()

 #           else:
 #               samples.extend(data)

 #           for x in data:
 #            print(data)
            samples.append(data)

        elif(started == True):
            print ("recording stopped")
            stream.stop_stream()

        #    hmm = random.choice(acknowledgements)
        #    playsound(hmm);

            recwavfile(samples, audio)

            #reset all vars
            started = False
            silence_buffer.clear()
            samples = []

            run = 0


    stream.close()
    audio.terminate()

    return AUDIO_FILE;


def recwavfile(data, audio):

#    print(data)
    wf = wave.open(AUDIO_FILE, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(data))
    wf.close()



def recoSpeech(filename):
    with sr.AudioFile(filename) as source:

        r = sr.Recognizer()
        r.energy_threshold = 50
        r.dynamic_energy_threshold = False

        audio = r.record(source)
        usertext = "";

        try:
            usertext = r.recognize_google(audio)

        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")

        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))


        return usertext






if __name__ == "__main__":

    playsound("global/media/cleanse.mp3", block=False)
    timestamp = str(int(time.time()))
    print(timestamp)
    
    r = sr.Recognizer()  # Initialize r here
    
    #check if there is an argument
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        # Read the text from the input file
        with open(filename, "r") as f:
            text = f.read()
            speaktext(text)
    else:
    
        wait_for_wake_word(r)
        get_mode_data()
        announce_session()
        chatony()
