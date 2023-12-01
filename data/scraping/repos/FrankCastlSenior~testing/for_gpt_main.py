import os
import openai
from openaiapi import get_chatgpt_response
from elevenlabsapi import ttsapikey
import io
import requests
from pydub import AudioSegment
from pydub.playback import play
from datetime import datetime
import webbrowser
import time
import threading

# Custom directory for saving text files
output_text_directory = "C:/Tic/main_gpt_assistant/output_chat"

# Eleven Labs API and Voice ID - "TmQmj1rrc2pDH2JOOfTi - Valley, NLhhnq7yGcjLD58e2Y83 - Neal"
ELEVEN_LABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech/NLhhnq7yGcjLD58e2Y83"
ELEVEN_LABS_API_KEY = ttsapikey


def text_to_speech(text):
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVEN_LABS_API_KEY,
    }

    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5,
            "use_speaker_boost": True,
        },
    }

    response = requests.post(ELEVEN_LABS_API_URL, json=data, headers=headers)

    audio = AudioSegment.from_mp3(io.BytesIO(response.content))
    play_audio_thread = threading.Thread(target=play, args=(audio,))
    play_audio_thread.start()  # Start playing audio in a separate thread


def save_response_to_file(prompt, response):
    current_time = datetime.now().strftime("%H:%M:%S")
    current_date = datetime.now().strftime("%Y-%m-%d")

    with open(
        os.path.join(output_text_directory, "output.txt"),
        "a",
        encoding="utf-8",
    ) as text_file:
        text_file.write(
            f"Prompt ({current_time}) ({current_date}) - {prompt}\nResponse - {response}\n\n"
        )


def open_website(url):
    webbrowser.open(url)


# text_to_speech("Welcome back, Captain")
print("Valley: Welcome back, Captain")

while True:
    user_input = input("Captain: ")
    user_input_lower = user_input.lower()

    if "valley sleep" in user_input_lower:
        text_to_speech("Goodbye, Captain.")
        time.sleep(1)
        print("Sam: Goodbye, Captain.")
        break

    response = get_chatgpt_response(user_input)

    play_audio = True

    # Check for specific commands and perform actions
    if "play music" in user_input_lower:
        musicPath = "C:/Users/dom/Music/Hymn_for_the_weekend.m4a"
        response = "Playing music, sir."
        text_to_speech(response)
        time_to_wait = 2
        play_audio = False
        time.sleep(time_to_wait)
        os.system(f"start {musicPath}")

    elif (
        "current time and date" in user_input_lower
        or "time and date" in user_input_lower
    ):
        current_time = datetime.now().strftime("%I:%M %p")
        current_date = datetime.now().strftime("%Y-%m-%d")
        response = f"The current time is {current_time} and the date is {current_date}."

    elif "current time" in user_input_lower or "time" in user_input_lower:
        current_time = datetime.now().strftime("%I:%M:%S %p")
        response = f"The current time is {current_time}."

    elif "current date" in user_input_lower or "date" in user_input_lower:
        current_date = datetime.now().strftime("%Y-%m-%d")
        response = f"The current date is {current_date}."

    sites = [
        ["youtube", "https://www.youtube.com"],
        ["wikipedia", "https://www.wikipedia.com"],
        ["google", "https://www.google.com"],
    ]

    for site in sites:
        if f"open {site[0]}".lower() in user_input_lower:
            response = f"Opening {site[0]} sir..."
            text_to_speech(response)
            time.sleep(2)
            play_audio = False
            open_website(site[1])

    if play_audio:
        text_to_speech(response)

    save_response_to_file(user_input, response)
    print("Valley:", response)
