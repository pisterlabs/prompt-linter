import speech_recognition as sr
import pyttsx3
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
import time
import openai
import json
from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
import threading
import itertools
import sys

# Initialize recognizer and the text to speech engine
r = sr.Recognizer()
engine = pyttsx3.init()

# Get the list of voices
voices = engine.getProperty('voices')

# Set the voice you want to use by its ID
desired_voice_id = voices[2].id  # Replace 0 with the index of the voice you want to use
engine.setProperty('voice', desired_voice_id)

# Set your OpenAI API key here
openai.api_key = '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'

# Load the conversation history from a file
try:
    with open('history.json', 'r') as f:
        messages = json.load(f)
except FileNotFoundError:
    # If the file doesn't exist, initialize the list of messages with the system message
    messages = [{"role": "system", "content": "You're name's Sofiya.You are a kinda bot, created by MK.You have the ability to remember personal details.You are created in Bagya Matriculation School.Your Developer Name is MK.He is an indie-Developer Who creates amazing games, applications, Animations, Websites, and some programmings.You're Created in Sri Bagya Matriculation School by MK.You must thankful for the teachers who supported in creating you.You're owner's name is MK.As an AI you have a name but not age or anything else.You are created esspeciealy for a Science Exhibition as his project.If you cant answer the question you just simply say, Ask MK.As well as you can speak even through spoken words but you support english only"}]

# Initializing the Edge driver
driver = webdriver.Edge(service=Service(r'C:\\Users\\Admin\\Downloads\\edgedriver_win64\\msedgedriver.exe'))

# Function to convert text to speech
def speak(text):
    print(text)  # Print the text to the console
    engine.say(text)
    engine.runAndWait()

# Function to listen to the microphone and recognize speech
def listen():
    global done
    done = False  # Reset the 'done' flag to False before starting listening

    t = threading.Thread(target=animate)  # Start the loading animation in a separate thread
    t.start()

    with sr.Microphone() as source:
        speak("Listening...")
        audio = r.listen(source)
        try:
            print("Processing...")
            text = r.recognize_google(audio)
            done = True  # Set 'done' to True after processing is finished to stop the loading animation
            speak("listened")
            print(f"You said : {text}")
            return text
        except:
            done = True  # Set 'done' to True after processing is finished to stop the loading animation
            speak("Sorry, I didn't get that")
            return listen()

# Function to play song on YouTube using Selenium
def play_song_on_youtube(song_name):
    # Navigate to the YouTube search page with the song name
    driver.get(f"https://www.youtube.com/results?search_query={song_name}")

    # Wait for the page to load
    time.sleep(2)

    # Find the first video and click it to play
    video = driver.find_element(By.ID, "video-title")
    video.click()
    skip_ad()

# Function to skip ad on YouTube using Selenium
def skip_ad():
    time.sleep(5)  # Wait for the ad to load
    try:
        # Find the "Skip Ads" button and click it
        skip_button = driver.find_element(By.CLASS_NAME, "ytp-ad-skip-button")
        skip_button.click()
        print("Ad skipped")
    except Exception as e:
        print("No skippable ad")

# Function to set system volume using pycaw library
def set_volume(volume):
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(ISimpleAudioVolume._iid_, 1, None)
    volume_interface = interface.QueryInterface(ISimpleAudioVolume)
    volume_interface.SetMasterVolume(volume, None)

# Function for loading animation while processing speech recognition
done = False  # Flag that indicates when processing is done

def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write('\rloading... please be patient' + c + ' ')
        sys.stdout.flush()
        time.sleep(0.1)

# Main functioning
def main():
    while True:
        command = listen()
        if command.lower() == "stop":
            break
        if "play" in command:
            song_name = command.split("play ", 1)[1]  # Get the song name from the command
            try:
                play_song_on_youtube(song_name)  # Open the search results for the specified song in the default web browser
                speak(f"Playing {song_name} on YouTube")
            except Exception as e:
                speak(f"Sorry, I couldn't play {song_name}")
        elif "set volume to" in command:
            volume_percentage = int(command.split("set volume to")[1].replace("%", "")) / 100.0  # Get volume percentage from command and convert it into a float between 0.0 and 1.0.
            try:
                set_volume(volume_percentage)  # Set system volume using pycaw library.
                speak(f"Setting volume to {volume_percentage * 100}%")
            except Exception as e:
                speak(f"Sorry, I couldn't set volume")
        else:
            # Add the user's question to the list of messages.
            messages.append({"role": "user", "content": command})

            # Get the answer from ChatGPT.
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages  # Use the list of messages here.
            )

            # Speak out and print the answer.
            answer = response['choices'][0]['message']['content']
            speak(answer)

            # Add the assistant's answer to the list of messages.
            messages.append({"role": "assistant", "content": answer})

            # Saving the conversation history to a file.
            with open('history.json', 'w') as f:
                json.dump(messages, f)

main()
