from dotenv import load_dotenv
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
from pydub import AudioSegment
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
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Load environment variables from .env file
load_dotenv()

# Access environment variables
openai.api_key = os.environ.get("OPENAI")
eleven_labs_api_key = os.environ.get("ELEVEN")

# Load the sounds
beep_start = 'media/beep_start.mp3'
beep_stop = 'media/beep_stop.wav'
tellme = 'media/tellme.mp3'
askme = 'media/askme.mp3'
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


########################################################################################
########  Initiate session  ########
########################################################################################

# create the seed
timestamp = str(int(time.time()))

# set the correct audio interface
def get_device_with_max_channels():
    """
    Find the device ID with the maximum number of output channels.

    Returns:
        int: Device ID with the highest number of output channels.
    """
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

# # print the result
# print("UNIX time is: " + {timestamp})


########################################################################################
########  Revelation - recording  ########
########################################################################################

revelation = "My name is Benjamin and I used to eat corn on the cob" # Initialize 'revelation' with an empty string

# # Initialize the recognizer
# r = sr.Recognizer()

# # Set the microphone as the source
# time.sleep(1)  # wait for 1 second
# playsound(tellme)
# with sr.Microphone() as source:

#     print("I'm listening...\n\n")
#     # Listen for speech and store it as audio data
#     audio = r.listen(source)
#     # Save the audio data to a WAV file
#     with open(f"working/recordings/{timestamp}_recording.wav", "wb") as f:
#         f.write(audio.get_wav_data())

# # Use the recognizer to convert speech to text, playing some atmospherics while we wait

# try:
#     def ambient_background(radiation, stop_event):
#         while not stop_event.is_set():
#             playsound(radiation)

#     # Create and start the thread
#     stop_event = threading.Event()
#     sound_thread = threading.Thread(target=ambient_background, args=(radiation, stop_event))
#     sound_thread.daemon = True
#     sound_thread.start()

#     # Recognize the speech input using Google Speech Recognition
#     playsound(acknowledgement)
#     text = r.recognize_google(audio)
#     print("\n\nI heard: " + text)

#     # Prepare to pass the transcript to the prompt
#     revelation = text
#     filename = f"working/revelations/{timestamp}_revelation.txt"

#     # Save the generated transcript to a file with epoch timestamp
#     with open(f"working/revelations/{timestamp}_revelation.txt", "w") as f:
#         f.write(revelation)

# except sr.UnknownValueError:
#     print("Google Speech Recognition could not understand audio")

# except sr.RequestError as e:
#     print("Could not request results from Google Speech Recognition service; {0}".format(e))

   
########################################################################################
########  Revelation — prompt  ########
########################################################################################

# The GPT-3.5 model ID you want to use
model_id = "text-davinci-003"

# The maximum number of tokens to generate in the response
max_tokens = 1024

# Extract the personality aspects from the revelation
prompt = f"I’m going to give you a story about me. Please interpret some subtle aspects of my story and what they reveal about my personality: {revelation}.\n\nRespond with a simple JSON array, strings only:"
response = openai.Completion.create(
    engine=model_id,
    prompt=prompt,
    max_tokens=max_tokens
)

# Save the personality aspects to a local file with an epoch timestamp
filename = f"working/analysis/{timestamp}_personality_aspects.txt"
with open(filename, "w") as f:
    f.write(response.choices[0].text)
personality_aspects = response.choices[0].text
print("\n\nPersonality aspects:" + personality_aspects)

# Come up with affirming responses
prompt = f"Respond to this story with a sympathetic one-liner, five different ways. You should make the speaker feel heard, seen and understood: \n\n'{revelation}.'\n\nRespond with a simple JSON array, strings only:"
# print("Prompt:\n\n" + prompt)
response = openai.Completion.create(
    engine=model_id,
    prompt=prompt,
    max_tokens=max_tokens
)

# Save the affirmations to a local file with an epoch timestamp
filename = f"working/affirmations/{timestamp}_affirmations.json"
with open(filename, "w") as f:
    f.write(response.choices[0].text)
affirmations = response.choices[0].text
print("\n\nAffirmations:" + affirmations)

	# TODO: fingerprint / search for fingerprint

	# TODO: if new, start voice cloning

########################################################################################
########  Affirmation — generation  ########
########################################################################################

voice = "https://api.elevenlabs.io/v1/text-to-speech/o7lPjDgzlF8ZloHzVPeK"
eleven_labs_api_key = "b74b7e7ae1605ee65d2f3e10145f54d0"  # replace with your API key

headers = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": eleven_labs_api_key
}

lock = Lock()

def send_to_api(text):
    global counter

    response = requests.post(voice, json={"text": text, "voice_settings": {"stability": 0, "similarity_boost": 0}}, headers=headers)
    if response.status_code == 200:
        with lock:  # Acquire lock before accessing counter
            filename = f"working/affirmations/{timestamp}_affirmation{counter}.mp3"
            counter += 1  # Safely increment counter
        
        with open(filename, "wb") as f:
            f.write(response.content)
            print(f"Generating audio for: {filename}")
        
        # Convert the MP3 to WAV
        wav_filename = convert_mp3_to_wav(filename)
        print(f"Converted to WAV: {wav_filename}")
        
        return filename
    else:
        print(f"Request to API failed with status code {response.status_code}.")
        return None


def main():
    global counter
    counter = 0  # Resetting the counter to 0 at the beginning of main

    with open(f"working/affirmations/{timestamp}_affirmations.json", "r") as file:
        texts = json.load(file)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(send_to_api, text) for text in texts}

if __name__ == "__main__":
    main()

########################################################################################
########  Affirmation — playback  ########
########################################################################################

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
    print(f"Routing {filename} to device {device_id} on channel {channel}")
    with sd.OutputStream(device=device_id, channels=max_output_channels, samplerate=sound.frame_rate) as stream:
        stream.write(zeros)

def list_files_with_same_timestamp(directory):
    # List all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Use regex to extract timestamp and counter from filenames
    pattern = r"(\d+)_affirmation(\d+).mp3"

    # Group files by timestamp
    timestamp_files = defaultdict(list)
    for file in files:
        match = re.match(pattern, file)
        if match:
            timestamp, counter = match.groups()
            timestamp_files[timestamp].append((int(counter), file))

    return timestamp_files


def main():
    directory = "working/affirmations"
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

        # Sleep for 1.25 seconds before starting the next thread
        time.sleep(1.25)

    # Wait for all threads to complete
    for t in threads:
        t.join()

if __name__ == "__main__":
    main()

########################################################################################
########  Request — recording  ########
########################################################################################

request = "ask me" # Initialize 'request' with an empty string

# Initialize the recognizer
r = sr.Recognizer()

# Set the microphone as the source
time.sleep(1)  # wait for 1 second
playsound(askme)
with sr.Microphone() as source:

    print("\n\nI'm listening...")
    # Listen for speech and store it as audio data
    audio = r.listen(source)
    # Save the audio data to a WAV file
    with open(f"working/requests/{timestamp}_request.wav", "wb") as f:
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
    print("\n\nI heard: " + text)

    # Prepare to pass the transcript to the prompt
    request = text
    filename = f"working/requests/{timestamp}_request.txt"

    # Save the generated transcript to a file with epoch timestamp
    with open(f"working/requests/{timestamp}_request.txt", "w") as f:
        f.write(request)
        print(f"\n\nRequest saved to working/requests/{timestamp}_request.txt")

except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")

except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))

########################################################################################
########  Request — prompt  ########
########################################################################################

# The GPT-3.5 model ID you want to use
model_id = "text-davinci-003"

# The maximum number of tokens to generate in the response
max_tokens = 1024

# Extract the personality aspects from the revelation
prompt = f"Your job is to help me psychoanalyze a fictional character. Here's a bit about the character: {personality_aspects}. Imagine they are asking the following introspective question: {request}. What is happening in their subconscious mind?"
response = openai.Completion.create(
    engine=model_id,
    prompt=prompt,
    max_tokens=max_tokens
)

# Save the personality aspects to a local file with an epoch timestamp
filename = f"working/analysis/{timestamp}_psychoanalysis.txt"
with open(filename, "w") as f:
    f.write(response.choices[0].text)
    print(f"\n\nOutput saved to working/analysis/{timestamp}_psychoanalysis.txt")
psychoanalysis = response.choices[0].text
print("\n\nPsychoanalysis:" + psychoanalysis)

# Come up with subconscious thoughts
prompt = f"Let's play a game. Pretend you are the following person: {personality_aspects}. You've asked yourself: {request}. Reflect on the following to discover at least five subconscious thoughts: {psychoanalysis}. \n\nRespond with a simple JSON array:"
print({prompt})
response = openai.Completion.create(
    engine=model_id,
    prompt=prompt,
    max_tokens=max_tokens
)

# Save the subconscious thoughts to a local file with an epoch timestamp
filename = f"working/subconsciousness/{timestamp}_subconcious.json"
with open(filename, "w") as f:
    f.write(response.choices[0].text)
    print(f"Subconscious thoughts saved to working/subconsciousness/{timestamp}_subconcious.json")
subconsciousness = response.choices[0].text
print("\n\nSubconscious thoughts:" + subconsciousness)

########################################################################################
########  Answer — generation  ########
########################################################################################

voice = "https://api.elevenlabs.io/v1/text-to-speech/o7lPjDgzlF8ZloHzVPeK"
eleven_labs_api_key = "b74b7e7ae1605ee65d2f3e10145f54d0"  # replace with your API key

headers = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": eleven_labs_api_key
}

lock = Lock()

def send_to_api(text):
    global counter

    response = requests.post(voice, json={"text": text, "voice_settings": {"stability": 0, "similarity_boost": 0}}, headers=headers)
    if response.status_code == 200:
        with lock:  # Acquire lock before accessing counter
            filename = f"working/subconsciousness/{timestamp}_subconcious{counter}.mp3"
            counter += 1  # Safely increment counter
        
        with open(filename, "wb") as f:
            f.write(response.content)
            print(f"Generating audio for: {filename}")
        
        # Convert the MP3 to WAV
        wav_filename = convert_mp3_to_wav(filename)
        print(f"Converted to WAV: {wav_filename}")
        
        return filename
    else:
        print(f"Request to API failed with status code {response.status_code}.")
        return None


def main():
    global counter
    counter = 0  # Resetting the counter to 0 at the beginning of main

    with open(f"working/subconsciousness/{timestamp}_subconcious.json", "r") as file:
        texts = json.load(file)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(send_to_api, text) for text in texts}

if __name__ == "__main__":
    main()


########################################################################################
########  Answer — playback  ########
########################################################################################

route_sound('working/subconsciousness/0_subconcious.wav', 2, 0)
route_sound('working/subconsciousness/1_subconcious.wav', 2, 0)
route_sound('working/subconsciousness/2_subconcious.wav', 2, 0)


# # ########  Request(n) ########  

# # 	# TODO: record audio, save, upload

# # 	# TODO: transcribe, save {request(n)}

# # 	PRINT: {request(n)}

# # 	PROMPT: Based on my {personality_aspects}, guess my motivations for {request(n)}. Your response should be only JSON.
# # 		RESPONSE: {motivations(n)}

# # 	PROMPT: You are to simulate aspects of my subconscious mind. Help me better understand myself; {personality_aspects}. Respond to {request(n)} in 5 ways. Consier my {motivations}. The 5 ways represent the following aspects of my personality: {mode}. Your response should be only JSON.
# # 		RESPONSE: {answer(n)}

# # 	PRINT: {motivations(n)} {answer(n)}

# # 	# TODO: generate audio for {answer(n)}

# # ########  Answer(n) ########  

# # 	AUDIO: 
# # 		1()
# # 		2()
# # 		3()
# # 		4()
# # 		5()

# # 	AUDIO: ask me another question x 10

# # ########  END PROGRAM ########  

# # # TODO: LOOP x n

# # # TODO: timeout

# # # TODO: save session, associate with user
