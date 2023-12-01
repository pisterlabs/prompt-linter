import io
from pydub import AudioSegment
from pydub.playback import play
import speech_recognition as sr
import whisper
import queue
import tempfile
import os
import threading
import click
import torch
import numpy as np
import time
import subprocess
import re
import pyaudio
from pathlib import Path
import struct
import openai
import requests
import pyclip
import json

from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

openai.api_key = os.getenv("OPENAI_TOKEN")
# print(openai.Model.list())
# exit(0)

@click.command()
@click.option("--model", default="small", help="Model to use", type=click.Choice(["tiny","base", "small","medium","large"]))
@click.option("--english", default=True, help="Whether to use English model",is_flag=True, type=bool)
@click.option("--verbose", default=False, help="Whether to print verbose output", is_flag=True,type=bool)
@click.option("--energy", default=300, help="Energy level for mic to detect", type=int)
@click.option("--dynamic_energy", default=False,is_flag=True, help="Flag to enable dynamic engergy", type=bool)
@click.option("--pause", default=1.8, help="Pause time before entry ends", type=float)
@click.option("--save_file",default=False, help="Flag to save file", is_flag=True,type=bool)
def main(model, english,verbose, energy, pause,dynamic_energy,save_file):
    temp_dir = tempfile.mkdtemp() if save_file else None
    #there are no english models for large
    if model != "large" and english:
        model = model + ".en"
    audio_model = whisper.load_model(model, device='cuda')
    audio_queue = queue.Queue()
    result_queue = queue.Queue()

    r = sr.Recognizer()
    r.energy_threshold = energy
    r.pause_threshold = pause
    r.dynamic_energy_threshold = dynamic_energy

    manager = ModelManager("models.json", progress_bar=True)
    model_path, config_path, model_item = manager.download_model("tts_models/en/ljspeech/tacotron2-DCA")

    vocoder_path, vocoder_config_path, _ = manager.download_model("vocoder_models/en/ljspeech/multiband-melgan")

    #synth = Synthesizer(model_path, config_path, vocoder_checkpoint=vocoder_path, vocoder_config=vocoder_config_path, use_cuda=True)
    synth = Synthesizer(model_path, config_path, use_cuda=True)

    keywords = {
        "gpt": chatgpt,
        "clippy": clippy
    }
    while True:
        audio_data = record_audio(r)
        print("before transcribe")
        texts = transcribe(audio_data, audio_model, english)
        print("after transcribe")
        print(texts["text"])
        for keyword, task in keywords.items():
            query = extract_query(texts["text"], keyword)
            if query is not None:
                text = task(query)
                
                print(text)

                if "clipboard" in query:
                    pyclip.copy(text)
                    text = "Copied to clipboard. " + text
                open("last_tts.txt", "w").write(text)
                p = subprocess.Popen(["mimic","-f", "last_tts.txt"])
                p.wait()


def chatgpt(prompt):
    # response = openai.Completion.create(model="gpt-4", prompt=prompt, temperature=0, max_tokens=100)
    # return response['choices'][0]['text'].strip()
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
        messages=[ 
            {"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": prompt}
        ])
#        , temperature=0, max_tokens=100)
    return response['choices'][0]['message']["content"].strip()

def clippy(prompt):
    res = requests.request("get", "http://localhost:8383", data=json.dumps(dict(prompt=prompt)))
    return res.text

def record_audio(r):
    #load the speech recognizer and set the initial energy threshold and pause threshold

    with sr.Microphone(sample_rate=16000) as source:
        print("Say something!")
        audio = r.listen(source)
        print("after audio")
        torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
        return torch_audio



def transcribe(audio_data, audio_model, english):
    if english:
        result = audio_model.transcribe(audio_data,language='english')
    else:
        result = audio_model.transcribe(audio_data)

    return result

def extract_query(text, keyword):
    parts = text.lower().split(keyword)
    if len(parts) > 1:        
        text = "".join(keyword.join(parts[1:]))
        text = re.sub(r'[^\w\s]', '', text)

        if len(text.strip()) > 0:
            return text
    return None

def tts(synth, text):
    wav = synth.tts(
        text,
    )
    synth.save_wav(wav, "tts.wav")
    p = subprocess.Popen(["aplay", "tts.wav"])
    p.wait()
        
main()
