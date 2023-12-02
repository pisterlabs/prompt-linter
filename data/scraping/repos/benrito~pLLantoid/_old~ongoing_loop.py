from dotenv import load_dotenv
import sys
import pyaudio
import wave
import time
import openai
import requests
import json
import os
import re
from audio_effects import OtherworldlyAudio
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
from scipy.io import wavfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import episode_manager

########################################################################################
########  Setup  ########
########################################################################################

# Load environment variables from .env file
load_dotenv()

# Access environment variables
openai.api_key = os.environ.get("OPENAI")
eleven_labs_api_key = os.environ.get("ELEVEN")

AGENT_NAME = "Plantony"
USER_NAME = "Human"


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
        prompt_text=open("./plantony.txt").read().strip(),
        user_name=USER_NAME,
        agent_name=AGENT_NAME,
        max_turns=5,
        end_phrases=[
            "hail gpt"
        ],
        prompt_config=default_prompt_config()
    ),
}

########################################################################################
########  Initiate session  ########
########################################################################################

# set the correct audio interface
def get_device_with_max_channels():
    devices = sd.query_devices()
    max_channels = 0
    max_device_id = None

    for device in devices:
        if device['max_output_channels'] > max_channels:
            max_channels = device['max_output_channels']
            max_device_id = device['index']

    if max_device_id is None:
        raise ValueError("No suitable device found.")
    
    return max_device_id

# load the current mode
def get_mode_data():
    try:
        with open('working/ongoing_config.json', 'r') as file:
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
timestamp = str(int(time.time()))
config_data = get_mode_data()
if config_data:
    for key, value in config_data.items():
        globals()[key] = value
if config_data and 'current_mode' in config_data:
    print(f"\n\nCurrent mode is: {config_data['current_mode']}")
else:
    print("Unable to retrieve current mode.")

# Load the sounds
base_path = f'modes/{current_mode}/media/'
ambient_files = [f'ambient{i}.mp3' for i in range(1, 5)]
acknowledgement_files = [f'acknowledgement{i}.mp3' for i in range(1, 5)]

ambient_sounds = [base_path + file for file in ambient_files]
acknowledgement_sounds = [base_path + file for file in acknowledgement_files]

radiation = random.choice(ambient_sounds)
acknowledgement = random.choice(acknowledgement_sounds)

print("\n\nInitiating session " + timestamp)

print("\n\nTap away")

########################################################################################
########  Main loop  ########
########################################################################################

def main_loop():
    def detect_taps(threshold=10, tap_interval=1, taps_required=3):
        """
        Continuously monitor the microphone for taps.
        """
        tap_count = 0
        last_tap_time = time.time()

        def callback(indata, frames, pa_time, status):
            nonlocal tap_count, last_tap_time
            volume_norm = np.linalg.norm(indata) * 10
            if volume_norm > threshold:
                print("\n\nTap detected!")  # Log message for tap detection
                if time.time() - last_tap_time < tap_interval:
                    tap_count += 1
                else:
                    tap_count = 1
                last_tap_time = time.time()

        with sd.InputStream(callback=callback):
            while tap_count < taps_required:
                sd.sleep(int(tap_interval * 1000))  # sleep for tap_interval seconds

        return True

    MAX_ATTEMPTS = 3
    attempts = 0

    r = sr.Recognizer()

    while attempts < MAX_ATTEMPTS:
        # Wait for taps
        if detect_taps():
            playsound(f"modes/church_of_plantony/media/beep_start.wav")

        time.sleep(1)  # wait for 1 second

        with sr.Microphone() as source:
            print("\n\nI'm listening...")
            audio = r.listen(source)
            with open(f"working/{current_mode}/recordings/{timestamp}_recording.wav", "wb") as f:
                f.write(audio.get_wav_data())

        # Use the recognizer to convert speech to text, playing some atmospherics while we wait
        try:

            def mirror_recording():
                mirror = OtherworldlyAudio(f"working/{current_mode}/recordings/{timestamp}_recording.wav")
                time.sleep(3)  # wait for n second
                mirror.pitch_shift()
                mirror.play_sound()

            def ambient_background(radiation, stop_event):
                while not stop_event.is_set():
                    playsound(radiation)

            # Create and start the thread
            stop_event = threading.Event()
            sound_thread = threading.Thread(target=ambient_background, args=(radiation, stop_event))
            mirror_thread = threading.Thread(target=mirror_recording)
            sound_thread.daemon = True
            sound_thread.start()

            # Recognize the speech input using Google Speech Recognition
            playsound(acknowledgement)
            mirror_thread.start()

            # TODO: add spinner
            text = r.recognize_google(audio)
            print("\n\nI heard: " + text)

            # Prepare to pass the transcript to the prompt
            revelation = text

            with open(f"working/{current_mode}/recordings/{timestamp}_recording.txt", "w") as f:
                f.write(revelation)

            break  # Exit the loop since we successfully got the transcript

        except sr.UnknownValueError:
            print("\n\nI couldn't understand you. Tap again.")
            playsound(f"modes/{current_mode}/media/beep_stop.wav")
            stop_event.set()  # Stop the radiation sound
            attempts += 1
            if attempts == MAX_ATTEMPTS:
                playsound(beep_stop)
                sys.exit(1)

        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            stop_event.set()  # Stop the radiation sound
            attempts += 1
            if attempts == MAX_ATTEMPTS:
                playsound(beep_stop)
                sys.exit(1)

    ########################################################################################
    ########  Revelation — prompt  ########
    ########################################################################################

    # The GPT-3.5 model ID you want to use
    model_id = "text-davinci-003"

    # The maximum number of tokens to generate in the response
    max_tokens = 1024

    # Extract the analysis from the recording
    prompt = f"{reasoning_prompt1}\n\n'{revelation}.'\n\nRespond with a simple JSON array, strings only:"
    # print("Prompt:\n\n" + prompt)
    response = openai.Completion.create(
        engine=model_id,
        prompt=prompt,
        max_tokens=max_tokens
    )

    # Save the analysis to a local file with an epoch timestamp
    filename = f"working/{current_mode}/analysis/{timestamp}_analysis.txt"
    with open(filename, "w") as f:
        f.write(response.choices[0].text)
    reasoning_prompt1_analysis = response.choices[0].text
    print(f"{reasoning_prompt1_msg}:\n\n{reasoning_prompt1_analysis}")

    # Come up with responses
    prompt = f"{collective_prompt1}\n\n'{revelation}.'\n\n'{reasoning_prompt1_analysis}.'\n\nRespond with a simple JSON array, strings only:"
    # print("Prompt:\n\n"prompt)
    response = openai.Completion.create(
        engine=model_id,
        prompt=prompt,
        max_tokens=max_tokens
    )

    # Save the responses to a local file with an epoch timestamp
    # TODO: avoid rewrites if not first time
    filename = f"working/{current_mode}/audio/{timestamp}_responses.json"
    with open(filename, "w") as f:
        f.write(response.choices[0].text)
    responses = response.choices[0].text
    print("\n\nResponses:" + responses)

        # TODO: fingerprint / search for fingerprint

        # TODO: if new, start voice cloning

    ########################################################################################
    ########  Audio response — generation  ########
    ########################################################################################

    voice = "https://api.elevenlabs.io/v1/text-to-speech/o7lPjDgzlF8ZloHzVPeK"
    # eleven_labs_api_key = "b74b7e7ae1605ee65d2f3e10145f54d0"  # replace with your API key

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": eleven_labs_api_key
    }

    lock = Lock()

    def send_to_eleven_labs_api(text):
        global counter
        
        # Send the responses to Eleven Labs API
        response = requests.post(voice, json={"text": text, "voice_settings": {"stability": 0, "similarity_boost": 0}}, headers=headers)
        if response.status_code == 200:
            with lock:  # Acquire lock before accessing counter
                filename = f"working/{current_mode}/audio/{timestamp}_response{counter}.mp3"
                counter += 1  # Safely increment counter
            
            with open(filename, "wb") as f:
                f.write(response.content)
                print(f"\n\nGenerating audio for: {filename}")
            
            # Convert the MP3 to WAV (will deprecate when we go to Bark)
            wav_filename = convert_mp3_to_wav(filename)
            print(f"Converted to WAV: {wav_filename}")
            
            return filename
        else:
            print(f"Request to API failed with status code {response.status_code}.")
            return None


    def main():
        global counter
        counter = 0  # Resetting the counter to 0 at the beginning of main

        with open(f"working/{current_mode}/audio/{timestamp}_responses.json", "r") as file:
            texts = json.load(file)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(send_to_eleven_labs_api, text) for text in texts}

    if __name__ == "__main__":
        main()

    ########################################################################################
    ########  Audio response — playback  ########
    ########################################################################################

    # Wait for the mirror sound to have played
    mirror_thread.join()

    def route_sound(filename, device_id, channel):

        # Load audio file using pydub
        sound = AudioSegment.from_file(filename, format="mp3")

        # Convert sound to numpy array and normalize
        samples = np.array(sound.get_array_of_samples()).astype(np.float32) / (2**15)
        
        max_output_channels = sd.query_devices(device_id)['max_output_channels']

        if channel >= max_output_channels:
            raise ValueError(f"The device only has {max_output_channels} output channel(s).")

        # Create an empty array with the correct number of output channels
        zeros = np.zeros((len(samples), max_output_channels), dtype=np.float32)

        # Copy the mono audio data to the desired channel
        zeros[:, channel] = samples

        # Start a stream and play it
        print(f"\n\nRouting {filename} to device {device_id} on channel {channel}")
        with sd.OutputStream(device=device_id, channels=max_output_channels, samplerate=sound.frame_rate) as stream:
            stream.write(zeros)

    def list_files_with_same_timestamp(directory):
        # List all files in the directory
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        
        # Use regex to extract timestamp and counter from filenames
        pattern = rf"(\d+)_response(\d+).mp3"

        # Group files by timestamp
        timestamp_files = defaultdict(list)
        for file in files:
            match = re.match(pattern, file)
            if match:
                timestamp, counter = match.groups()
                timestamp_files[timestamp].append((int(counter), file))

        return timestamp_files


    def main():
        directory = f"working/{current_mode}/audio"
        device_id = get_device_with_max_channels()  # Automatically get the device ID with the most channels
        target_timestamp = timestamp

        timestamp_files = list_files_with_same_timestamp(directory)
        
        if target_timestamp not in timestamp_files:
            print(f"No files with timestamp {target_timestamp} found.")
            return

        files = timestamp_files[target_timestamp]
        
        # Sort the files based on counter for a deterministic order
        sorted_files = sorted(files, key=lambda x: x[0])
        
        max_output_channels = sd.query_devices(device_id)['max_output_channels']

        threads = []  # To store thread objects

        for index, (counter, filename) in enumerate(sorted_files):
            # Cycle through the number of available channels
            channel = index % max_output_channels

            # Create a thread for each sound and start it immediately
            t = threading.Thread(target=route_sound, args=(os.path.join(directory, filename), device_id, channel))
            threads.append(t)
            t.start()

            # Sleep for n seconds before starting the next thread
            time.sleep(2)

        # Wait for all threads to complete
        for t in threads:
            t.join()

        print("Tap again...")

    if __name__ == "__main__":
        main()

if __name__ == "__main__":
    try:
        while True:
            main_loop()
    except KeyboardInterrupt:  # Press Ctrl+C to exit the loop
        print("\nScript terminated by user.")
