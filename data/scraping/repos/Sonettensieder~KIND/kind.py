import tkinter as tk
import sounddevice as sd
import pyttsx3
import openai
from scipy.io.wavfile import write
import os
import json
import numpy as np

# GLOBALS

recording = None
engine = pyttsx3.init()
fs = 44100  # Sample rate
settings = {
    "api_key": "",
    "role": "",
   # "seconds": 5
}

# Load Settings
def load_settings():
    if os.path.exists("settings.json") == True:
        file = open("settings.json", "r")
        settings.clear()
        newsettings = json.load(file)
        settings.update(newsettings)
        file.close()    

load_settings()
openai.api_key = settings["api_key"]

# Save Settings
def save_settings():
    settings["api_key"] = api_entry.get()
    settings["role"] = role_entry.get("1.0", "end")
   # settings["seconds"] = int(seconds_entry.get())
    file = open("settings.json", "w")
    json.dump(settings, file)
    file.close()

# Record
def start_recording():
    global recording, stream
    recording = []
    record_button.config(text="Aufnahme beenden", command=stop_recording)
    stream = sd.InputStream(callback=callback)
    stream.start()

def stop_recording():
    global recording, stream
    record_button.config(state="disabled")    
    stream.stop()
    stream.close()
    write("output.wav", fs, np.concatenate(recording))
    recording = None
    ask_kind()

def callback(indata, frames, time, status):
    global recording
    if status:
        # print(status, file=sys.stderr)
        return
    if np.any(indata):
        recording.append(indata.copy())

# Send to ChatGPT
def ask_kind():
    audio_file = open("output.wav", "rb")
    response = openai.Audio.transcribe("whisper-1", audio_file)

    # Assume the response returns the transcribed audio in text format
    transkribiertes_audio = response["text"]
    # print ("Transkript: "+transkribiertes_audio) # For debug only

    # Send the transcribed audio to ChatGPT
    bot_answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": settings["role"]},
        {"role": "user", "content": transkribiertes_audio},
        ]
    )

    message = bot_answer.choices[0].message.content
    # print ("Antwort KI.ND: " + message) # For debug only

    # Play the response from ChatGPT
    play_response(message, transkribiertes_audio)
    
    # Delete the audio file
    # os.remove("output.wav") # commented out because line threw error considering writing rights

    record_button.config(state="normal", text="Aufnahme", command=start_recording)

def play_response(response, transkribiertes_audio):
    output_label = tk.Label(master_frame, wraplength=480, width=81, height=22, text="Antwort KI.ND: " + response, anchor="nw", justify="left", font=(None, 8))
    output_label.grid(column=0, row=6, columnspan=4, sticky="w")
    transcript_label = tk.Label(master_frame, wraplength=390, width=60, height=2, text="Transkript: " + transkribiertes_audio, anchor="w", justify="left", font=(None, 8))
    transcript_label.grid(column=1, row=5, columnspan=3, sticky="w")
    root.update_idletasks()
    engine.say(response)
    engine.runAndWait()

# GUI
root = tk.Tk()
root.title("KI.ND v1.2")
root.geometry("538x666")

master_frame = tk.Frame(root, borderwidth=25,bg="#66CDAA")
master_frame.grid(column=0, row=0)
master_frame.rowconfigure(0, pad=8)
master_frame.rowconfigure(1, pad=8)
master_frame.rowconfigure(2, pad=0)
master_frame.rowconfigure(3, pad=0)
master_frame.rowconfigure(4, pad=20)
master_frame.rowconfigure(5, pad=20)
master_frame.rowconfigure(6, pad=8)
master_frame.rowconfigure(7, pad=8)
# master_frame.columnconfigure(0, weight = 0)
# master_frame.columnconfigure(1, weight = 1)
# master_frame.columnconfigure(2, weight = 2)
# master_frame.columnconfigure(3, weight = 2)

api_entry_label = tk.Label(master_frame, text="API-Key:")
api_entry_label.grid(column=0, row=0, sticky="w")

api_entry = tk.Entry(master_frame, width=60)
api_entry.insert(0, settings["api_key"])
api_entry.grid (column=1, row=0, columnspan=3, sticky = "w")

# seconds_entry_label = tk.Label(master_frame, text="Aufnahmedauer (Sek):")
# seconds_entry_label.grid(column=0, row=1)

# seconds_entry = tk.Entry(master_frame, width=3)
# seconds_entry.insert(0, settings["seconds"])
# seconds_entry.grid(column=1, row=1, sticky = "w")

role_entry_label = tk.Label(master_frame, text="Hier die Rolle eingeben, welche KI.ND annehmen soll. Das kann eine freie Beschreibung sein,\n welche Identität, Weltanschauung, Beruf und alles mögliche weitere enthalten kann:")
role_entry_label.grid(column=0, columnspan=4, row=2)

role_entry = tk.Text(master_frame, height=6, width=61, wrap="word")
role_entry.insert("1.0", settings["role"])
role_entry.grid(column=0, columnspan=4, row=3)

save_button = tk.Button(master_frame, text="Eingaben Speichern", command=save_settings)
save_button.grid(column=0, row=4, sticky = "w")

record_button = tk.Button(master_frame, text="Aufnahme", command=start_recording, state="normal")
record_button.grid(column=0, row=5, sticky = "w")

transcript_label = tk.Label(master_frame, wraplength=390, width=60, height=2, text="Transkript: ", anchor="w", justify="left", font=(None, 8))
transcript_label.grid(column=1, row=5, columnspan=3, sticky="w")

output_label = tk.Label(master_frame, wraplength=480, width=81, height=22, text="Antwort KI.ND:\n", anchor="nw", justify="left", font=(None, 8))
output_label.grid(column=0, row=6, columnspan=4, sticky="w")

root.mainloop()