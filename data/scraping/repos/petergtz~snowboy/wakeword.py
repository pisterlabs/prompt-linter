#!/usr/bin/env python3

import audioop
import os
import subprocess
import wave
from datetime import datetime

import boto3
import openai

import pyaudio
import requests
from pydub import AudioSegment
from pydub.playback import play
import simpleaudio as sa
import snowboydecoder
from snowboydecoder import no_alsa_error
from AudioLib import AudioEffect

# Parameters for audio recording
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024


openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

robot_name = "Robby"

messages = [{
    "role": "system",
    "content": f"You're the brain of a little robot called {robot_name}. " +
               "It's a toy for kids and you should behave like a toy. Be funny. "
    #   +"In addition, with every response you give, please prepend it with "+
    #   "one of the following intents, that best match to the users intent: "+
    #   "THANKS_INTENT, COME_INTENT, BYE_INTENT, GO_FORWARD_INTENT, GO_BACKWARD_INTENT, OTHER_INTENT"
}]


def detected_callback():
#    detector.terminate()
    print("Wake word detected. Recording...", end="", flush=True)
    while True:
        frames = []

        with no_alsa_error():
            audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, input_device_index=1, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        i = 0
        max_level_volume = 0
        last_samples = [0] * 20
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

            rms = audioop.rms(data, 2)  # 2 is the width of the sample (in bytes)
            last_samples = last_samples[1:] + [rms]
            average = sum(last_samples) / len(last_samples)
            max_level_volume = max(max_level_volume, average)
            # print("max rms", max_level_volume)
            # print("last_sample", last_samples)
            if i > int(RATE / CHUNK * 2) and average < max_level_volume / 20:
                break

            i += 1

        print("finished.")

        stream.stop_stream()
        stream.close()
        audio.terminate()

        with wave.open("recording.wav", 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

        text = transcribe()
        print("Transcript:", text)
        messages.append({
            "role": "user",
            "content": f"{text}\n"
        })

        text = submit_to_chatgpt(messages)
        messages.append({
            'role': 'assistant',
            'content': text,
        })

        text_to_speech(text)

        print("Converting to wav using ffmpeg...", end="")
        start_time = datetime.now()
        subprocess.check_call(["ffmpeg", "-i", "response.mp3", "-y", "response.wav"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(datetime.now() - start_time)
        print("Making it robotic...", end="")
        start_time = datetime.now()
        AudioEffect.robotic("response.wav", 'response_robotic.wav')
        print(datetime.now() - start_time)

#        audio = AudioSegment.from_file("response.mp3", format="mp3")
#        with no_alsa_error():
#            play(audio)
#            play(audio.speedup(playback_speed=1.3).set_frame_rate(22050))
        sa.WaveObject.from_wave_file("response_robotic.wav").play().wait_done()


polly = boto3.client('polly')


def text_to_speech(text):
    print("Transcribing with Polly...", end="")

    start_time = datetime.now()
    response = polly.synthesize_speech(
        Text=text,
        OutputFormat="mp3",
        VoiceId='Vicki',
        SampleRate='22050',
    )
    print(datetime.now() - start_time)

    with open("response.mp3", 'wb') as file:
        file.write(response['AudioStream'].read())


def submit_to_chatgpt(messages):
    print("Sending to ChatGPT...", end="")

    start_time = datetime.now()

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    print(datetime.now() - start_time)
    text = response['choices'][0]['message']['content']
    print("Response:", text)

    return text


def transcribe():
    files = {"file": ("recording.wav", open("recording.wav", "rb")), }

    print("Transcribing...", end="")
    start_time = datetime.now()
    response = requests.post("https://api.openai.com/v1/audio/transcriptions",
                             headers={"Authorization": f"Bearer {openai_api_key}"},
                             files=files,
                             data={"model": "whisper-1"})
    print(datetime.now() - start_time)

    if response.status_code != 200:
        raise Exception("Unexpected response:" + str(response))

    return response.json()['text']

detected_callback()

# detector = snowboydecoder.HotwordDetector(['resources/models/computer.umdl'], sensitivity=0.5)
detector = snowboydecoder.HotwordDetector(['robby-shure.pmdl'], sensitivity=5.5)

print(f"Listening for the wake word '{robot_name}'... Press Ctrl+C to exit.")
detector.start(detected_callback)
detector.terminate()
