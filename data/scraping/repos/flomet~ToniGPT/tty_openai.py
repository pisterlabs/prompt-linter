# -*- coding: utf-8 -*-

from openai import OpenAI
from pathlib import Path




def tty(client:OpenAI, txt: str, model:str = "tts-1", voice:str = "alloy", mp3_file:Path = Path("speech.mp3"))->Path:
    
    response = client.audio.speech.create(
        model = model,
        voice = voice.lower(), 
        input = txt
    )
    response.stream_to_file(mp3_file)
    return mp3_file


if __name__ == "__main__":
    client = OpenAI()
    text ="Dies ist ein Beispieltext von mir"
    mp3_file = tty(client, text)

