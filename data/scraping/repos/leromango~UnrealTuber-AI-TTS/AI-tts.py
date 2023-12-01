import openai
import requests
import time
from azure.cognitiveservices.speech import SpeechConfig, SpeechRecognizer, AudioConfig
import speech_recognition as sr
import keyboard
from threading import Event
import obsws_python as obs
import os
import json
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

def load_config():
    global obsPassword, obsPort, obsItem, apiKeyElevenLabs, apiKeyOpenAI, ttsVoiceID, voiceStability, voiceSimilarity, personality, phraseTimeLimit, activationKey, chosenDeviceIndex, srLevel, apiKeyAzure, apiKeyAzureRegion
    # Load the configuration file
    with open(config_file_path, "r") as file:
        config = json.load(file)

    # Assign the loaded config values to the variables in your project
    obsPassword = config["WebSocket password from OBS: "]
    obsPort = config["WebSocket port from OBS: "]
    obsItem = config["Item name to be controlled in OBS: "]
    apiKeyElevenLabs = config["ElevenLab API Key: "]
    apiKeyOpenAI = config["OpenAI API Key: "]
    ttsVoiceID = config["ElevelLabs Voice ID: "]
    voiceStability = config["Voice Stability settings for ElevenLabs API: "]
    voiceSimilarity = config["Voice Similarity settings for ElevenLabs API: "]
    personality = config["Character personality: "]
    activationKey = config["Key to activate the bot: "]
    chosenDeviceIndex = config["Microphone device index: "]
    srLevel = config["Use Azure's speech recognition? (1 = yes, 0 = no): "]
    apiKeyAzure = config["Azure API Key: "]
    apiKeyAzureRegion = config["Azure API Region: "]
    phraseTimeLimit = config["If not using Azure, enter phrase time limit: "]

#Record voice, transcribe it, get ChatGPT response
def start_recording_simple(e):
    global is_playing
    print("Please start speaking...")
    with sr.Microphone(device_index=chosenDeviceIndex) as source:
        try:
            audio = r.listen(source, phraseTimeLimit)
        except sr.WaitTimeoutError:
            print("Timeout error: didn't hear anything for the duration.")
            return

        print("Finished speaking. Processing...")
        try:
            result = r.recognize_google(audio)
            print("You said: " + result)
            if result != "":
                is_playing = True
                send_user_input(None, result)
            else:
                print("No speech detected in the recording.")
        except:
            print("Sorry, I did not get that. Please try again.")
            return

# Variable to track whether we're currently recording
is_recording = False
recording = None
is_playing = False

def start_recording():
    global is_recording, recording
    if not is_recording:
        print("Started recording")
        recording = sd.rec(int(10 * 44100), samplerate=44100, channels=2, device=chosenDeviceIndex)
        is_recording = True

def end_recording():
    # Set up the Azure speech recognizer
    speech_config = SpeechConfig(subscription=str(apiKeyAzure), region=str(apiKeyAzureRegion))
    global is_recording, recording, is_playing
    if is_recording:
        print("Recording finished. Recognizing...")
        sd.wait()  # Wait for the recording to finish
        is_recording = False
        is_playing = True
        if np.all(recording == 0):
            print("No sound detected in the recording.")
            return

        try:
            write('input.wav', 44100, recording)  # Save recording to a WAV file
        except Exception as e:
            print(f"Error while writing the recording: {e}")
            return

        # Transcribe the WAV files
        try:
            audio_config = AudioConfig(filename='input.wav')
            recognizer = SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
            result = recognizer.recognize_once()
            print("Recognized: {}".format(result.text))
            if result.text != "":
                send_user_input(None, result.text)
            else:
                print("No speech detected in the recording.")

        except Exception as e:
            print(f"Error while recognizing the speech: {e}")
    else:
        print("Can't start recording while playing audio.")

def send_user_input(e, user_input):
    characterInfo = (
        personality + (
            """
            DON'T USE EMOJIS IN YOUR SPEECH! 
            DON'T ASK QUESTIONS UNDER ANY CIRCUMSTANCE! 
            KEEP YOUR SPEECH SHORT! MAXIMUM OF 3-5 SENTENCES!
            AVOID USING COMPLEX VOCABULARY!
            Try to always refer to yourself in first person!
            """
        )
    )
    #Call OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=[
            {"role": "system", "content": characterInfo
            },
            {"role": "user", "content": user_input},
        ]
    )

    responseText = response['choices'][0]['message']['content']
    print(responseText)
    tts(responseText)

#Send API call to Eleven Labs to voice generated text
def tts(textinput):
    global is_playing
    # Convert input text to string
    text = str(textinput)
    # Set TTS model ID and voice settings
    model_id = "eleven_monolingual_v1"
    voice_settings = {
        "stability": voiceStability,
        "similarity_boost": voiceSimilarity
    }

    # Set headers and data for TTS API request
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": apiKeyElevenLabs
    }
    data = {
        "text": text,
        "model_id": model_id,
        "voice_settings": voice_settings
    }

    # Send TTS API request and save response to file
    response = requests.post(url, json=data, headers=headers)
    output_file = "output.mp3"
    with open(output_file, "wb") as f:
        f.write(response.content)

    # Load and play TTS audio using Pygame
    pygame.mixer.init()
    pygame.mixer.music.load(output_file)
    pygame.mixer.music.play()

    # Enable OBS scene item and wait for TTS audio to finish playing
    current_scene = str(cl.get_current_program_scene().current_program_scene_name)
    itemID = cl.get_scene_item_id(current_scene, obsItem).scene_item_id
    cl.set_scene_item_enabled(current_scene, itemID, True)
    while pygame.mixer.music.get_busy():
        continue

    # Stop and unload Pygame mixer and hide OBS scene item
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()
    time.sleep(1)
    current_scene = str(cl.get_current_program_scene().current_program_scene_name)
    itemID = cl.get_scene_item_id(current_scene, obsItem).scene_item_id
    cl.set_scene_item_enabled(current_scene, itemID, False)
    is_playing = False

config_file_path = "config.json"

# Check if the configuration file exists
if os.path.exists(config_file_path):
    print("Config file found!")
    load_config()
else:
    print("Config file not found!")
    
    obsPassword = input ("Input your OBS password: ")
    obsPort = input ("Input your OBS port: ")
    obsItem = input("Input the name of an item to be controlled in OBS: ")
    apiKeyOpenAI = input("Input your OpenAI API key: ")
    apiKeyElevenLabs = input("Input your ElevenLabs API key: ")
    
    input("Press Enter to continue to select a desired voice...")
    url = "https://api.elevenlabs.io/v1/voices"
    headers = {
        "Accept": "application/json",
        "xi-api-key": apiKeyElevenLabs
    }
    #  Retrieve the list of available voices
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    voices = []
    for voice in response.json()["voices"]:
        voices.append({voice["name"]: voice["voice_id"]})
    # Print the available voices with each voice and its ID on a separate line
    print("Available voices:")
    for voice in voices:
        for name, voice_id in voice.items():
            print(f"{name}: {voice_id}")
    
    ttsVoiceID = input("Input ElevenLabs Voice ID: ")

    voiceStability = input("Input your prefered voice stability (0-1): ")
    voiceSimilarity = input("Input your prefered voice similarity (0-1): ")
    personality = input("Input your character personality: ")

    srLevel = int(input("Use Azure's speech recognition? (1 = yes, 0 = no): "))
    if srLevel == 0:
        phraseTimeLimit = int(input("Input how long should AI listen to silence before ending the recording (Recommended: 5): "))
    else:
        phraseTimeLimit = 0
        apiKeyAzure = input("Input your Azure API key: ")
        apiKeyAzureRegion = input("Input your Azure API region: ")

    activationKey = str(input("Select activation key: ")) #Choose a button to summon bot
    
    # Mic selection
    input("Press Enter to continue to select your microphone...")
    mic_list = sr.Microphone.list_microphone_names()
    for idx, microphone_name in enumerate(mic_list): # Displays all microphones
        print(f"Microphone with ID {idx} is {microphone_name}")
    chosenDeviceIndex = int(input("Select microphone index: "))
    print(f"The default microphone name is: {mic_list[chosenDeviceIndex]}")

    print("Settings saved!")

    # Prompt the user to input their settings
    config = {
        "WebSocket password from OBS: ": obsPassword,
        "WebSocket port from OBS: ": obsPort,
        "Item name to be controlled in OBS: ": obsItem,
        "ElevenLab API Key: ": apiKeyElevenLabs,
        "OpenAI API Key: ": apiKeyOpenAI,
        "Azure API Key: ": apiKeyAzure,
        "Azure API Region: ": apiKeyAzureRegion,
        "ElevelLabs Voice ID: ": ttsVoiceID,
        "Voice Stability settings for ElevenLabs API: ": voiceStability,
        "Voice Similarity settings for ElevenLabs API: ": voiceSimilarity,
        "Character personality: ": personality,
        "Key to activate the bot: ": activationKey,
        "Microphone device index: ": chosenDeviceIndex,
        "Use Azure's speech recognition? (1 = yes, 0 = no): ": srLevel,
        "If not using Azure, enter phrase time limit: ": phraseTimeLimit
    }


    # Save the configuration file
    with open(config_file_path, "w") as file:
        json.dump(config, file, indent=4, separators=(",\n", ": "))

openai.api_key = apiKeyOpenAI
cl = obs.ReqClient(host='localhost', port=obsPort, password=obsPassword, timeout=3)
url = f"https://api.elevenlabs.io/v1/text-to-speech/{ttsVoiceID}"

if srLevel == 1:
    print(f"Press '{activationKey}' to start speaking, and then press it again to stop.")
    keyboard.add_hotkey(activationKey, lambda: start_recording() if not is_recording and not is_playing else end_recording())
else:
    print(f"Press '{activationKey}' to start speaking, and the recording will end automatically after {phraseTimeLimit} seconds of silence.")
    keyboard.add_hotkey(activationKey, lambda: recording.set() if not recording.is_set() and not is_playing else print("Can't start recording while recording is in progress!"))

r = sr.Recognizer()

recording = Event()

while True:
    if srLevel == 0:
        recording.wait()  
        recording.clear()  
        start_recording_simple(recording)
    time.sleep(0.1)  # to prevent excessive CPU usage