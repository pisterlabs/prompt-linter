#!/bin/python
import os
import socket
from openai import OpenAI
import sounddevice as sd
from scipy.io.wavfile import write

openai_client = OpenAI(
    api_key = os.environ["OPENAI_API_KEY"]
)

# Setup socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("127.0.0.1", 1234))
welcome = s.recv(1024)
print(welcome.decode())

try:
    # Audio recording parameters
    fs = 44100  # Sample rate
    seconds = 3  # Duration

    while True:

        # audio recording
        input("Press enter to start recording...")
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()
        write('output.mp3', fs, myrecording)  # Save as WAV file

        # audio transcription
        audio_file= open("output.mp3", "rb")
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            response_format="text"
        )

        print(transcript)

        s.sendall(transcript.encode())

        server_response = s.recv(2048)
        print(server_response.decode())

finally:
    s.close()

# (2023.11.30__01.34.47) 
# (2023.11.30__02.20.46) MVP done. I should be able to do this quicker, esp if I'm not exhausted lol.