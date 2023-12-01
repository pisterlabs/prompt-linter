import cohere
import requests
import os
import threading
import time

API_KEY_COHERE = '' #API KEY FROM COHERE API (LANGUAGE MODEL)
API_KEY_ELEVEN = '' #API KEY FROM ELEVENLABS API (SPEECH MODEL)
def print_with_delay(text):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(0.04)
    print()

def play_audio():
    os.system('ffplay -nodisp -autoexit -loglevel quiet output.mp3')
    time.sleep(5)
    os.remove('output.mp3')

while True:
    message = input("\nYOU 👨‍💻: ")
    co = cohere.Client(API_KEY_ONE)
    response = co.generate(
        model='command-xlarge-beta',
        prompt=message,
        max_tokens=300,
        temperature=0.9,
        k=0,
        p=0.75,
        stop_sequences=[],
        return_likelihoods='NONE'
    )
    ai_response = response.generations[0].text
    url = 'https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM'
    headers = {
        'accept': 'audio/mpeg',
        'xi-api-key': API_KEY_ELEVEN,
        'Content-Type': 'application/json'
    }
    data = {
        'text': ai_response,
        'voice_settings': {
            'stability': 0,
            'similarity_boost': 0
        }
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        with open('output.mp3', 'wb') as f:
            f.write(response.content)
            audio_thread = threading.Thread(target=play_audio)
            audio_thread.start()
            print_with_delay('AI 🤖: {}'.format(ai_response))
    else:
        print('Text-to-speech request failed:', response.text)
