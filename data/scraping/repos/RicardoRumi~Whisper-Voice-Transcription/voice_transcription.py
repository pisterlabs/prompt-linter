import pyaudio
import wave
import threading
from pynput import keyboard
from openai import OpenAI
import pyperclip
import pyautogui
import os

import socket
import sys

def create_socket_lock(port=52245):
    """ Try to bind to a socket on localhost at the specified port. """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('127.0.0.1', port))
        return s
    except socket.error:
        sys.exit("Another instance of the script is running.")

lock_socket = create_socket_lock()
API_KEY_FILE = 'openai_api_key.txt'

def get_api_key():
    return "CREDENTIAL"

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
FILE_NAME = "recording.wav"

is_recording = False
recording_thread = None

audio = pyaudio.PyAudio()

def start_recording():
    global is_recording
    is_recording = True
    print("Recording started")

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    frames = []

    while is_recording:
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()

    wave_file = wave.open(FILE_NAME, 'wb')
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(audio.get_sample_size(FORMAT))
    wave_file.setframerate(RATE)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()

    transcribe_audio()

def transcribe_audio():
    print("Transcribe audio called")
    api_key = get_api_key()
    client = OpenAI(api_key=api_key)
    with open(FILE_NAME, "rb") as audio_file:
        transcript_response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )

    if hasattr(transcript_response, 'text'):
        transcript = transcript_response.text
        print(transcript)
        pyperclip.copy(transcript)
        print("Transcription copied to clipboard.")
        pyautogui.hotkey('ctrl', 'v')
    else:
        print("Transcription failed or the response format is not as expected.")

def stop_recording():
    global is_recording
    is_recording = False
    print("Recording stopped")

def on_press(key):
    global is_recording, recording_thread
    try:
        if key == keyboard.Key.cmd_r and not is_recording:
            recording_thread = threading.Thread(target=start_recording)
            recording_thread.daemon = True
            recording_thread.start()
    except AttributeError:
        pass

def on_release(key):
    global is_recording
    try:
        if key == keyboard.Key.cmd_r  and is_recording:
            stop_recording()
    except AttributeError:
        pass

def main():
    try:
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
    except KeyboardInterrupt:
        print("Exiting script...")
        stop_recording()

if __name__ == "__main__":
    main()
