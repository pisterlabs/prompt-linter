import numpy as np
import openai
import sounddevice as sd
from scipy.io import wavfile
import tempfile
import subprocess
from gtts import gTTS
import os, sys
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.agents import load_tools

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from core.commands import (computer_applescript_action,
            chrome_open_url,
            chrome_get_the_links_on_the_page,
            chrome_read_the_page,
            chrome_click_on_link)

# load environment variables
load_dotenv()
class VoiceAssistant:
    def __init__(self):
        # Set your OpenAI API key
        api_key = os.environ['OPENAI_API_KEY']
        openai.api_key = api_key
        # Initialize the assistant's history
        self.history = [
            {
                "role": "system",
                "content": "You are a helpful assistant. The user is English. Only speak English.",
            }
        ]
        llm = OpenAI(temperature=0, openai_api_key=api_key) # type: ignore 
        tools = [
            computer_applescript_action,
            chrome_open_url,
            chrome_get_the_links_on_the_page,
            chrome_read_the_page,
            chrome_click_on_link
        ]
        self.agent = initialize_agent(tools, llm, initialize_agent="zero-shot-react-description", verbose=True)

    def listen(self):
        """
        Records audio from the user and transcribes it.
        """
        print("Listening...")
        # Record the audio
        duration = 5  # Record for 3 seconds
        fs = 44100  # Sample rate

        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
        sd.wait()

        # # Save the NumPy array to a temporary wav file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
            wavfile.write(temp_wav_file.name, fs, audio)

            # Use the temporary wav file in the OpenAI API
            transcript = openai.Audio.transcribe("whisper-1", temp_wav_file)

        print(f"User: {transcript['text']}")
        return transcript['text']

    def think(self, text):
        """
        Generates a response to the user's input.
        """
        # Add the user's input to the assistant's history
        self.history.append({"role": "user", "content": text})

        # Use the agent to generate a response
        response = self.agent.run(text)

        if isinstance(response, str):
            # Handle the case when the response is a string
            message = response
        else:
            # Extract the assistant's response from the agent output
            message = response["content"]

        self.history.append({"role": "system", "content": message})
        print("Assistant:", message)
        return message


    def speak(self, text):
         # Convert text to speech using gTTS
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save the speech as a temporary audio file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts.save(temp_audio.name)
        
        # Play the audio file through the speakers
        subprocess.run(["afplay", temp_audio.name])

    def run(self):
        while True:
            print("""Initialising ChatGPT and Text-To-Speech...\n""")
            print("Welcome to IntelliVoiceAI! How can I assist you? If unsure say help")

            text = self.listen()
            formattedText = text.strip().lower()

            if "goodbye" in formattedText or "bye" in formattedText:
                print("Assistant: Goodbye! Have a great day!")
                self.speak("Goodbye! Have a great day!")
                break

            if "list" in formattedText or "note" in formattedText: 
                from src.skills.todo_list import todoList
                todolist = todoList(self)
                todolist.create_todo_list()
                break

            if "speed" in formattedText or "internet speed" in formattedText: 
                from src.skills.internet_test import InternetSpeed, SpeedHistory
                history_file_path = "speed_history.json"
                speed_history = SpeedHistory(history_file_path)
                speed = InternetSpeed(self, speed_history)
                speed.run()
                            
            if "weather" in formattedText:
                from src.skills.weather import Weather
                weather = Weather(self, None)
                weather.run()
                
            if "exit" in formattedText or "quit" in formattedText:
                print("Goodbye")
                break

            response = self.think(text)
            self.speak(response)
