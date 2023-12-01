# the following program is based on code provided by DevMiser - https://github.com/DevMiser

import datetime
import io
import openai
import os
import pvcobra
import pvporcupine
import pyaudio
import random
import socket
import struct
import schedule
import sys
import threading
import time
import traceback
import urllib.request

from colorama import Fore, Style
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance
from pvrecorder import PvRecorder #, create
from threading import Thread, Event
from time import sleep

import RPi.GPIO as GPIO
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
led1_pin = 4
GPIO.setup(led1_pin, GPIO.OUT)
GPIO.output(led1_pin, GPIO.LOW)

audio_stream = None
cobra = None
pa = None
porcupine = None
recorder = None
wav_file = None

openai.api_key = "put your secret API key between these quotation marks"
pv_access_key = "put your secret access key between these quotation marks"

Clear_list = [
    "Clear",
    "Clear the screen",
    "Clear the display",
    "Clear the canvas",
    "Delete",
    "Clean",
    "Clean the screen",
    "Clean the display",
    "Clean the canvas",
    "Wipe",
    "Wipe the screen",
    "Wipe the display",
    "Wipe the canvas",
    "Erase",
    "Erase the screen",
    "Erase the display",
    "Erase the canvas",
    "Blank screen",
    "Blank Display"
]

def clean_screen():
    cycles = 2
    colours = (display.RED, display.BLACK, display.WHITE, display.CLEAN)
    colour_names = (display.colour, "red", "black", "white", "clean")
    img = Image.new("P", (display.WIDTH, display.HEIGHT))
    for i in range(cycles):
        print("Cleaning cycle %i\n" % (i + 1))
        for j, c in enumerate(colours):
            print("- updating with %s" % colour_names[j+1])
            display.set_border(c)
            for x in range(display.WIDTH):
                for y in range(display.HEIGHT):
                    img.putpixel((x, y), c)
            display.set_image(img)
            display.show()
            time.sleep(1)
        print("\n")
    print("Cleaning complete")

def current_time():
    time_now = datetime.datetime.now()
    formatted_time = time_now.strftime("%m-%d-%Y %I:%M %p\n")
    print("The current date and time is:", formatted_time)

def dall_e2(prompt):
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="512x512" # Can also be 256x256 or 1024x1024
        )
        return (response['data'][0]['url'])
    except ConnectionResetError:
        print("ConnectionResetError")
        current_time()

def detect_silence():
    cobra = pvcobra.create(access_key=pv_access_key)
    silence_pa = pyaudio.PyAudio()
    cobra_audio_stream = silence_pa.open(
                    rate=cobra.sample_rate,
                    channels=1,
                    format=pyaudio.paInt16,
                    input=True,
                    frames_per_buffer=cobra.frame_length)
    last_voice_time = time.time()
    while True:
        cobra_pcm = cobra_audio_stream.read(cobra.frame_length)
        cobra_pcm = struct.unpack_from("h" * cobra.frame_length, cobra_pcm)
        if cobra.process(cobra_pcm) > 0.2:
            last_voice_time = time.time()
        else:
            silence_duration = time.time() - last_voice_time
            if silence_duration > 1.3:
                print("End of request detected\n")
                GPIO.output(led1_pin, GPIO.LOW)
                cobra_audio_stream.stop_stream
                cobra_audio_stream.close()
                cobra.delete()
                last_voice_time = None
                break

def fade_leds(event):
    pwm1 = GPIO.PWM(led1_pin, 200)

    event.clear()

    while not event.is_set():
        pwm1.start(0)
        for dc in range(0, 101, 5):
            pwm1.ChangeDutyCycle(dc)
            time.sleep(0.05)
        time.sleep(0.75)
        for dc in range(100, -1, -5):
            pwm1.ChangeDutyCycle(dc)
            time.sleep(0.05)
        time.sleep(0.75)

def listen():
    cobra = pvcobra.create(access_key=pv_access_key)
    listen_pa = pyaudio.PyAudio()
    listen_audio_stream = listen_pa.open(
                rate=cobra.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=cobra.frame_length)
    print("Listening...")
    while True:
        listen_pcm = listen_audio_stream.read(cobra.frame_length)
        listen_pcm = struct.unpack_from("h" * cobra.frame_length, listen_pcm)
        if cobra.process(listen_pcm) > 0.3:
            print("Voice detected")
            listen_audio_stream.stop_stream
            listen_audio_stream.close()
            cobra.delete()
            break

def refresh():
    print("\nThe screen refreshes every day at midnight to help prevent burn-in\n")
    current_time()
    clean_screen()
    sleep(5)
    print("\nRe-rendering")
    display.set_image(img_resized)
    #    display.set_border(display.BLACK)
    display.show()
    print("\nDone")

def refresh_schedule(event2):
    schedule.every().day.at("00:00").do(refresh)
    event2.clear()
    while not event2.is_set():
        schedule.run_pending()
        sleep(1)

def wake_word():
    porcupine = pvporcupine.create(keywords=["computer", "jarvis", "Art-Frame"],
                            access_key=pv_access_key,
                            sensitivities=[0.1, 0.1, 0.1], #from 0 to 1.0 - a higher number reduces the miss rate at the cost on increased false alarms
                                   )
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)
    wake_pa = pyaudio.PyAudio()
    porcupine_audio_stream = wake_pa.open(
                    rate=porcupine.sample_rate,
                    channels=1,
                    format=pyaudio.paInt16,
                    input=True,
                    frames_per_buffer=porcupine.frame_length)
    Detect = True
    while Detect:
        porcupine_pcm = porcupine_audio_stream.read(porcupine.frame_length)
        porcupine_pcm = struct.unpack_from("h" * porcupine.frame_length, porcupine_pcm)
        porcupine_keyword_index = porcupine.process(porcupine_pcm)
        if porcupine_keyword_index >= 0:
            GPIO.output(led1_pin, GPIO.HIGH)
            print(Fore.GREEN + "\nWake word detected\n")
            current_time()
            print("What would you like me to render?\n")
            porcupine_audio_stream.stop_stream
            porcupine_audio_stream.close()
            porcupine.delete()
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
            Detect = False

class Recorder(Thread):
    def __init__(self):
        super().__init__()
        self._pcm = list()
        self._is_recording = False
        self._stop = False

    def is_recording(self):
        return self._is_recording

    def run(self):
        self._is_recording = True

        recorder = PvRecorder(device_index=-1, frame_length=512)
        recorder.start()

        while not self._stop:
            self._pcm.extend(recorder.read())
        recorder.stop()

        self._is_recording = False

    def stop(self):
        self._stop = True
        while self._is_recording:
            pass

        return self._pcm

try:
    #o = create(
        #access_key=pv_access_key,
        #enable_automatic_punctuation=False,
    #)

    event = threading.Event()
    event2 = threading.Event()

    while True:
        wake_word()
        event2.set()
        recorder = Recorder()
        recorder.start()
        listen()
        detect_silence()
        transcript, words = o.process(recorder.stop())
        t_fade = threading.Thread(target=fade_leds, args=(event,))
        t_fade.start()
        recorder.stop()
        if transcript not in Clear_list:
            current_time()
            prompt_full = transcript
            print("You requested: " + prompt_full)
            print("\nCreating...")
            image_url = dall_e2(prompt_full)
            raw_data = urllib.request.urlopen(image_url).read()
            img = Image.open(io.BytesIO(raw_data))
            img_bordered = ImageOps.expand(img, border=(76, 0), fill='black')
            img_resized = img_bordered.resize((600, 448), Image.ANTIALIAS)
            print("\nRendering...")
            display.set_image(img_resized)
            img.show()
            event.set()
            GPIO.output(led1_pin, GPIO.LOW)
            sleep(2)
            t_refresh = threading.Thread(target=refresh_schedule, args=(event2,))
            t_refresh.start()
            print("\nDone")
        else:
            print("Clearing the display...")
            clean_screen()
            event.set()
            event2.set()
            GPIO.output(led1_pin, GPIO.LOW)
            print("\nDone")

except ConnectionResetError:
    print("Reset Error")
    current_time()

except KeyboardInterrupt:
    exit()
