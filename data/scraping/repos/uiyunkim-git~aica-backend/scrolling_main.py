#! python3.7
import requests
from queue import Queue
from time import sleep
import openai
import threading
import pyaudio
import tempfile
import wave
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost",  # Allow requests from this origin
    "http://localhost:3000",  # Allow requests from this origin and specific port
    "https://example.com",  # Allow requests from this specific origin
    "https://example.com:8080",  # Allow requests from this specific origin and port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)

DEEPL_API_KEY = '54b2312e-e3e9-1334-418e-bbce189c4b90:fx'
openai.api_key = "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"

stt_buffer = ""
stt_buffer_cropped = ""
audio_data_buffer = []
audio_data_buffer_crop = []
translated_text = ""

FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1               # Number of audio channels (1 for mono, 2 for stereo)
RATE = 44100               # Sample rate (samples per second)
CHUNK = 1024                # Number of frames per buffer

p = pyaudio.PyAudio()
def record():
    stop_recording = threading.Event()
    
    def audio_callback(in_data, frame_count, time_info, status):

        # start = end
        in_data_copy = in_data[:]

        audio_data_buffer.extend(in_data_copy)
        audio_data_buffer_crop.extend(in_data_copy)

        # print(len(audio_data_buffer))
        # if len(audio_data_buffer) > 44100 * CHUNK *  1:
        #     audio_data_buffer = audio_data_buffer[len(audio_data_buffer) - 44100 * 200:]
        
        return (in_data, pyaudio.paContinue)

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=audio_callback)

    stream.start_stream()

    def stop_recording_thread():
        input("Press Enter to stop recording...\n")
        stop_recording.set()
        record_thread.join()
        stream.stop_stream()
        stream.close()
        p.terminate()

    record_thread = threading.Thread(target=stop_recording_thread)
    record_thread.start()

def speech_to_text():
    def stt_thread():
        global stt_buffer
        global stt_buffer_cropped
        while True:
            stt_buffer = get_text()
            print("STT: ",stt_buffer)
            print("FULL: ",stt_buffer_cropped + ' ' + stt_buffer)
            
            sleep(1.2)

    stt_thread = threading.Thread(target=stt_thread)
    stt_thread.start()


def translate():
    def translate_thread():
        global translated_text
        global stt_buffer_cropped
        global stt_buffer
        while True:
            translated_text = translate_text(stt_buffer_cropped + ' ' + stt_buffer, "EN")
            print("Translate: ",translated_text)

    translate_thread = threading.Thread(target=translate_thread)
    translate_thread.start()

def transcribe():
    text = None
    while True:
        new_text = get_text(audio_data_buffer)
        if text != new_text:
            text =  new_text
            translated_text = translate_text(text, "EN")
            print(text,translated_text)


def get_text():
    global audio_data_buffer_crop
    global stt_buffer_cropped
    try:
        with wave.open("temp.wav", 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(RATE)
            wf.writeframes(bytearray(audio_data_buffer_crop))
            
        with open("temp.wav","rb") as f:
            response = openai.audio.transcriptions.create(
                model="whisper-1",
                file= f,
                response_format="verbose_json"
            )
        text = response
        if len(text.segments) >= 4:
            end_timestamp = text.segments[len(text.segments) - 4]["end"]
            buffer_size_to_delete = int(RATE * 2 * end_timestamp)
            audio_data_buffer_crop = audio_data_buffer_crop[buffer_size_to_delete:]
            for seg in text.segments[:len(text.segments) - 3]:
                stt_buffer_cropped = stt_buffer_cropped + ' ' + seg["text"]

        # return text.segments
        return text.text
    except Exception as e:
        print(e)
        return e

def translate_text(text, target_language):
    deepl_url = 'https://api-free.deepl.com/v2/translate'
    params = {
        'text': text,
        'target_lang': target_language,
        'auth_key': DEEPL_API_KEY,
    }
    response = requests.post(deepl_url, data=params)

    translation_data = response.json()
    translations = translation_data.get('translations', [])

    if translations:
        return translations[0].get('text', '')
    
def join_str_common_prefix_substring(str1, str2):
    common_substring = ""
    
    min_length = min(len(str1), len(str2))
    
    # Iterate through characters from the end of str1 and the beginning of str2
    for i in range(1, min_length + 1):
        suffix = str1[-i:]
        prefix = str2[:i]
        
        # Check if the suffix of str1 matches the prefix of str2
        if suffix == prefix:
            common_substring = suffix
    
    return str1.strip(common_substring) +' '+ str2

@app.get("/")
def get_stt_buffer():
    global stt_buffer
    global stt_buffer_cropped
    global translated_text
    return {
            'data':
                {
                    'original_text':stt_buffer,
                    'translated_text':translated_text,
                    'third_text':stt_buffer_cropped + ' ' + stt_buffer
                }
            }




record()
time.sleep(2)
speech_to_text()
translate()