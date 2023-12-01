import os
import struct
import wave
import pvporcupine
from pvrecorder import PvRecorder
import pyttsx3
import threading
from oswaveplayer import *
import speech_recognition as sr
import time
from datetime import datetime
import subprocess
import openai

openai.api_key =  "YOUR_API_KEY_HERE"
openai.api_base = "https://chimeragpt.adventblocks.cc/api/v1"



def get_headers():
    return {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}






r = sr.Recognizer()
print("Starting Otto...")
voice = sr.AudioFile('voice.wav')
with voice as source:
    audio = r.record(source)
    command = r.recognize_google(audio)
    print("Otto version: 1.0.0")

def play_voice():
    engine.runAndWait()

engine = pyttsx3.init()

access_key = "I6tODUeq7aHSiLYui53UpoIUNIE/DdtKuVOTRE0U47MkUoSIvtlqEQ=="
keyword_paths = ["/home/pi/hey-Otto_en_raspberry-pi_v2_2_0.ppn"]
sensitivities = [0.5] * len(keyword_paths)

try:
    porcupine = pvporcupine.create(
        access_key=access_key,
        keyword_paths=keyword_paths,
        sensitivities=sensitivities
    )

    keywords = [os.path.basename(x).replace('.ppn', '').split('_')[0] for x in keyword_paths]

    print('Porcupine version: %s' % porcupine.version)

    recorder = PvRecorder(frame_length=porcupine.frame_length, device_index=-1)
    recorder.start()

    wav_file = None
    output_path = None

    if output_path is not None:
        wav_file = wave.open(output_path, "w")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)

    print('Listening ... (press Ctrl+C to exit)')

    while True:
        pcm = recorder.read()
        result = porcupine.process(pcm)

        if wav_file is not None:
            wav_file.writeframes(struct.pack("h" * len(pcm), *pcm))

        if result >= 0:
            print('[%s] Detected %s' % (str(datetime.now()), keywords[result]))
            yourSound = playwave("Wakewordnoise.wav")
            engine.say(' . ')
            threading.Thread(target=play_voice).start()

            import pyaudio
            def record_audio(filename, duration=5, sample_rate=44100, chunk_size=1024, format_=pyaudio.paInt16, channels=2):
                audio = pyaudio.PyAudio()
                stream = audio.open(format=format_, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk_size)
                print("Recording...")
                frames = []
                for i in range(0, int(sample_rate / chunk_size * duration)):
                    data = stream.read(chunk_size)
                    frames.append(data)
                print("Recording finished.")
                stream.stop_stream()
                stream.close()
                audio.terminate()
                wave_file = wave.open(filename, "wb")
                wave_file.setnchannels(channels)
                wave_file.setsampwidth(audio.get_sample_size(format_))
                wave_file.setframerate(sample_rate)
                wave_file.writeframes(b''.join(frames))
                wave_file.close()

            output_filename = "voice.mp3"
            recording_duration = 5
            record_audio(output_filename, duration=recording_duration)
            voice = sr.AudioFile('voice.mp3')
            with voice as source:
                audio = r.record(source)
            command = r.recognize_google(audio)

            user_messages = [
                {"role": "system", "content": "You are a AI voice assistant called Otto."},
                {"role": "user", "content": command}
            ]


            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo-16k',
                messages=user_messages,
                stream=True
            )

            ai_response = ""


            print("Otto:", ai_response)

            engine.say(ai_response)
            threading.Thread(target=play_voice).start()



except KeyboardInterrupt:
   print('Stopping ...')
finally:
    recorder.delete()
    porcupine.delete()
    if wav_file is not None:
        wav_file.close()
