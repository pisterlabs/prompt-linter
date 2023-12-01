#!/usr/bin/env python3
#Porcupine wakeword includes
import struct
import pyaudio
import pvporcupine
import wave
import time
import speech_recognition as sr
import subprocess
import openai
import json

OUTPUT_FILE = '/tmp/output.wav'
with open('config.json','r') as h:
    keys = json.load(h)
#in case you want to use the openai api for whisper
openai.api_key = keys['openai_key']
porcupine = None
pa = None
audio_stream = None

porcupine = pvporcupine.create(access_key=keys['porcupine_api_key'],keywords=['jarvis'], sensitivities=[0.7])
try:

    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
                    rate=porcupine.sample_rate,
                    channels=1,
                    format=pyaudio.paInt16,
                    input=True,
                    frames_per_buffer=porcupine.frame_length)
    
    while True:
        pcm = audio_stream.read(porcupine.frame_length)
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
        keyword_index = porcupine.process(pcm)

        if keyword_index >= 0:
            print("wake word detected")
            counter = time.time()
            received_data = bytearray()
            while True:
                pcm = audio_stream.read(porcupine.frame_length)
                received_data.extend(pcm)
                if(time.time() - counter > 5):
                    break
                # Save the received audio to a file
            with wave.open(OUTPUT_FILE, 'wb') as audio_file:
                audio_file.setnchannels(1)  # Mono
                audio_file.setsampwidth(2)  # 16-bit
                audio_file.setframerate(16000)  # Sample rate
                audio_file.writeframes(received_data)
            with open(OUTPUT_FILE,'rb') as h:
                transcript = openai.Audio.transcribe("whisper-1",file=h)
            text = transcript['text']
            print(text)
            args = ['./assistant.py',text]
            process = subprocess.Popen(args, stdout=subprocess.PIPE)
            output, error = process.communicate()
            print(output)
            print(error)
            code = process.wait()
except Exception as e:
    print(e)