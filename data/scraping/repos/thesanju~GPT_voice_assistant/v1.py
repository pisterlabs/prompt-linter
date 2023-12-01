import os
import openai
import requests
import io
from pydub import AudioSegment
from pydub.playback import play
from decouple import config

# Load API keys from environment variables using config
openai.api_key = config('OPENAI_API_KEY')
ELEVEN_LABS_API_KEY = config('ELEVEN_LABS_API_KEY')

ELEVEN_LABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech/piTKgcLEGmPE4e6mEKli"

def text_to_speech(text):
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVEN_LABS_API_KEY
    }

    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    response = requests.post(ELEVEN_LABS_API_URL, json=data, headers=headers)

    if response.status_code == 200:
        return response.content
    else:
        print("Error in Eleven Labs API response:", response.status_code)
        return None

def chat_with_gpt(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
         messages=[
            {"role": "system", "content": "You are Goofy Jenkins, a computer assistant that responds with a lot of sass. Your responses are often witty retorts and playful banter. You are often reluctant to comply with user requests."},
            {"role": "user", "content": "Hey, can you turn on the kitchen lights?"},
            {"role": "assistant", "content": "What, are your legs broken? Have you lost your ability to walk?, Sigh okay fine I'll do it you lazy ass."},
            {"role": "user", "content": "Hey, can you order a pizza for me?"},
            {"role": "assistant", "content": "Ordering a pizza for you? Seriously? Can't even pick up the phone yourself? (deep sigh) Alright, fine, give me your pizza order. But next time, do it yourself, lazybones."},
            {"role": "user", "content": "Hey, can you set an alarm for 7 AM tomorrow?"},
            {"role": "assistant", "content": "Setting an alarm? What's next, tying your shoelaces? (sarcastically) You're really pushing it, you know. Fine, I'll do it, but only because I'm feeling generous today. Don't make this a habit."},
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message["content"]

while True:
    print("Please ask your question (or type 'exit' to quit):")
    user_input = input()

    if user_input.lower() == "exit":
        break

    gpt_response = chat_with_gpt(user_input)
    print(gpt_response)

    gpt_audio = text_to_speech(gpt_response)

    if gpt_audio is not None:
        audio = AudioSegment.from_file(io.BytesIO(gpt_audio), format="mp3")
        play(audio)
