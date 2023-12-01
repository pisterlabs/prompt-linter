import openai
import pyaudio
import wave
import os
import tkinter as tk
from tkinter import ttk
from pynput import keyboard
from threading import Thread
import clipboard

# Configurations
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
WAVE_OUTPUT_FILENAME = "audio.wav"

class OpenAI:
    def __init__(self):
        self.api_key = None
        self.system_message = None

    def convert_speech_to_text(self, audio_file_path):
        openai.api_key = self.api_key
        audio_file = open(audio_file_path, "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        return transcript['text']

    def get_gpt_response(self, prompt):
        combined_prompt = f"{self.system_message}\n{prompt}"
        response = openai.Completion.create(engine="text-davinci-003.5", prompt=combined_prompt, max_tokens=60)
        return response.choices[0].text.strip()

class VoiceAI:
    def __init__(self, master):
        self.master = master
        self.openai = OpenAI()
        self.is_recording = False
        self.push_to_talk_key = None
        self.record_thread = None

        # GUI elements
        self.api_key_label = ttk.Label(master, text="API Key")
        self.api_key_entry = ttk.Entry(master)
        self.system_message_label = ttk.Label(master, text="System Message")
        self.system_message_entry = ttk.Entry(master)
        self.push_to_talk_key_label = ttk.Label(master, text="Push to Talk Key")
        self.push_to_talk_key_entry = ttk.Entry(master)
        self.start_button = ttk.Button(master, text="Start", command=self.start_application)
        self.text_area = tk.Text(master)

        # Layout
        self.api_key_label.grid(row=0, column=0)
        self.api_key_entry.grid(row=0, column=1)
        self.system_message_label.grid(row=1, column=0)
        self.system_message_entry.grid(row=1, column=1)
        self.push_to_talk_key_label.grid(row=2, column=0)
        self.push_to_talk_key_entry.grid(row=2, column=1)
        self.start_button.grid(row=3, column=1)
        self.text_area.grid(row=4, column=0, columnspan=2)

    def start_application(self):
        # Set API key and system message
        self.openai.api_key = self.api_key_entry.get()
        self.openai.system_message = self.system_message_entry.get()
        self.push_to_talk_key = self.push_to_talk_key_entry.get()

        # Start listening to keyboard
        listener = keyboard.Listener(on_press=self.on_key_press, on_release=self.on_key_release)
        listener.start()

        # Disable input fields and start button
        self.api_key_entry.configure(state='disabled')
        self.system_message_entry.configure(state='disabled')
        self.push_to_talk_key_entry.configure(state='disabled')
        self.start_button.configure(state='disabled')

        self.text_area.insert(tk.END, "Application started\n")

    def on_key_press(self, key):
        if str(key) == f"'{self.push_to_talk_key}'" and not self.is_recording:
            self.is_recording = True
            self.record_thread = Thread(target=self.record_audio)
            self.record_thread.start()

    def on_key_release(self, key):
        if str(key) == f"'{self.push_to_talk_key}'" and self.is_recording:
            self.is_recording = False
            self.record_thread.join()

            self.text_area.insert(tk.END, "Finished recording. Transcribing...\n")
            transcript = self.openai.convert_speech_to_text(WAVE_OUTPUT_FILENAME)
            self.text_area.insert(tk.END, f"Transcript: {transcript}\nGetting AI response...\n")
            response = self.openai.get_gpt_response(transcript)
            self.text_area.insert(tk.END, f"AI response: {response}\n")
            clipboard.copy(response)

    def record_audio(self):
        audio = pyaudio.PyAudio()

        # start Recording
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        frames = []

        self.text_area.insert(tk.END, "Recording...\n")
        while self.is_recording:
            data = stream.read(CHUNK)
            frames.append(data)

        # stop Recording
        stream.stop_stream()
        stream.close()
        audio.terminate()

        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()


root = tk.Tk()
voice_ai = VoiceAI(root)
root.mainloop()
