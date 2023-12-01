""" Text to speech helper functions """
import os
import requests
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Keys
openai.api_key = os.getenv("OPENAI_KEY2")
openai.organization = os.getenv("OPENAI_ORG2")
playht_api_key = os.getenv("PLAY_HT_KEY")
playht_user_id = os.getenv("PLAY_HT_ID")

async def text_to_audio(full_response):
    """ Convert text to streaming audio """

    url = "https://api.play.ht/api/v2/tts/stream"

    payload = {
        "text": f"{full_response}",
        "voice": "Harold",
        "output_format": "mp3",
        "voice_engine": "PlayHT2.0-turbo"
    }
    headers = {
        "accept": "audio/mpeg",
        "content-type": "application/json",
        "AUTHORIZATION": f"{playht_api_key}",
        "X-USER-ID": f"{playht_user_id}"
    }

    response = requests.post(url, json=payload, headers=headers, tiemout=10)

    print(response.text)
