from openai import OpenAI
import openai
from . import model, auth
from retry import retry
import streamlit as st
from pathlib import Path
from io import BytesIO
from threading import Thread
from time import time
from functools import lru_cache

client = OpenAI(api_key=auth.get_openai_key())
accepted_types = ['wav', 'mp3', 'mp4', 'm4a', 'webm']
task_params = {
    model.Task.ASR.value: {
        'model': 'whisper-1',
        'url': 'https://api.openai.com/v1/engines/whisper-1/completions',
        'max_tokens': 244,
    },
    model.Task.TTS.value: {
        'model': 'tts-1-hd',
        'url': 'https://api.openai.com/v1/audio/speech',
        'max_tokens': 2000,
    }
}

@retry(tries=3, delay=1)
def transcript(audio_file, prompt=None):
    """Transcript audio file to text"""
    try:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file, 
            prompt=prompt
        )
    except Exception as e:
        print(e)
        st.error(e)
    return transcript['text']


# @retry(tries=3, delay=1)
@lru_cache(maxsize=1000)
def text_to_speech(input_text, output_file=None, voice='nova', speed=1.2) -> BytesIO:
    # Supported voices are alloy, echo, fable, onyx, nova, and shimmer
    if not output_file: # 'must provide either `output_file` or `play_audio`'
        output_file = BytesIO()
    config = task_params[model.Task.TTS.value]
    
    try:
        response = client.audio.speech.create(
            model=config['model'],
            voice=voice,
            input=input_text,
            response_format='mp3',
            speed=speed
        )
    except openai.RateLimitError as e:
        st.error(e)
    t = time()
    if isinstance(output_file, str):
        response.stream_to_file(output_file)
    elif isinstance(output_file, BytesIO):
        for chunk in response.iter_bytes():
            output_file.write(chunk)
    print(f'saving took {time()-t} seconds')
    return output_file
    
# play in the background
def play_tts(input_text, voice='nova', speed=1.2):
    t = Thread(target=text_to_speech, kwargs={'input_text':input_text, 'play_audio':True, 'voice':voice, 'speed':speed})
    t.daemon = True
    t.start()
    

if __name__ == '__main__':
    text = '华为自研的其他功能模块对华为手机等设备有着直接而深远的影响。华为不仅自研了鸿蒙系统，还在芯片领域取得了一定的成果，这为其手机模块的性能、功能提升以及5G领域的技术创新提供了有力支撑。同时，自研芯片可以填补国产手机在关键技术上的空缺，提高市场竞争力。另外，自研芯片可以缩短物联网设备间的信息传输距离，提高传输速度，同时使得生产成本得到有效控制。'
    text_to_speech(text)