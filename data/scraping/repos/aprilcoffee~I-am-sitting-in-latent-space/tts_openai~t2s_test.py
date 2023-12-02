
import base64
import time
import openai
import os
import requests
import config 
import io

from pydub import AudioSegment
from pydub.playback import play

def speech(prompt):
    response = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers={
            "Authorization": f"Bearer {config.openai_api_key}",
        },
        json={
            "model": "tts-1",
            "input": prompt,
            "voice": "shimmer",
        },
    )

    audio = b""
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        audio += chunk

    # Create an AudioSegment from the audio bytes
    audio_segment = AudioSegment.from_file(io.BytesIO(audio), format="mp3")

    # Play the audio
    play(audio_segment)


    # Save the audio as a .wav file
    audio_segment.export("output.wav", format="wav")


