import streamlit as st
import time
import openai
import pyaudio
import wave
import numpy as np
from pydub import AudioSegment
from sentences import Sentences

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
SILENCE_THRESHOLD = 1500
SILENCE_FRAMES_THRESHOLD = 10  # Adjust the number of consecutive silent frames to detect silence
RECORDING_DELAY = 2



def record_audio():
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("start recording...")

    frames = []
    consecutive_silent_frames = 0
    is_recording = True
    start_time = time.time()
    while is_recording:
        data = stream.read(CHUNK)
        frames.append(data)

            # Convert the audio data to a numpy array
        samples = np.frombuffer(data, dtype=np.int16)

        # Calculate the energy level (volume) of the recorded audio data
        energy = np.max(np.abs(samples))
        print("energy level: " + str(energy))

        if time.time() - start_time >= RECORDING_DELAY:
            if energy < SILENCE_THRESHOLD:
                consecutive_silent_frames += 1
                if consecutive_silent_frames >= SILENCE_FRAMES_THRESHOLD:
                    is_recording = False
            else:
                consecutive_silent_frames = 0

    print("recording stopped")

    stream.stop_stream()
    stream.close()
    p.terminate()

    #create a wav file
    wf = wave.open("input.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close

    # convert_wav_to_mp3(wav_path="input.wav", mp3_path="input.mp3")

    print("done")
    # openai.api_key = "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"
    openai.api_key = "audio"
    audio_file = open("input.wav", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file, language="en")
    print(transcript.text)
    if(len(transcript.text) == 0):
        Sentences.set_popupmsg("System Advice: Please Speak into the Microphone")
    else:
        Sentences.set_sentence(transcript.text)
        
    