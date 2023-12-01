import os
import uuid

import openai
import requests
import soundfile as sf

from config import config

openai.api_key = config.OPENAI_API_KEY

def chat_completion(prompt: str) -> str:
    try:
        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        )

        return completion['choices'][0]['message']['content']
    except:
        return config.ERROR_MESSAGE
    
def transcript_audio(media_url: str) -> dict:
    try:
        ogg_file_path = f'{config.OUTPUT_DIR}/{uuid.uuid1()}.ogg'
        data = requests.get(media_url)
        with open(ogg_file_path, 'wb') as file:
            file.write(data.content)
        audio_data, sample_rate = sf.read(ogg_file_path)
        mp3_file_path = f'{config.OUTPUT_DIR}/{uuid.uuid1()}.mp3'
        sf.write(mp3_file_path, audio_data, sample_rate)
        audio_file = open(mp3_file_path, 'rb')
        os.unlink(ogg_file_path)
        os.unlink(mp3_file_path)
        transcript = openai.Audio.transcribe(
            'whisper-1', audio_file, api_key=config.OPENAI_API_KEY)
        return {
            'status': 1,
            'transcript': transcript['text']
        }
    except Exception as e:
        print('Error at transcript_audio...')
        print(e)
        return {
            'status': 0,
            'transcript': transcript['text']
        }
    