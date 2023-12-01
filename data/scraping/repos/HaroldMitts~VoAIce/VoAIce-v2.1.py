import tkinter as tk
import threading
import queue
import json
import os
import subprocess
import tempfile
import time
import openai
import azure.cognitiveservices.speech as speechsdk

from tkinter import filedialog
from queue import Queue

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()

        with open("voices.json", "r") as f:
            voice_data = json.load(f)

        self.voices = [voice["DisplayName"] for voice in voice_data]
        self.voice_map = {voice["DisplayName"]: voice["ShortName"] for voice in voice_data}

        self.current_voice = tk.StringVar(self.master)
        self.current_voice.set("en-US-SaraNeural")

        self.gpt_model = tk.StringVar(value="gpt-3.5-turbo")  # default model

        self.max_tokens = tk.IntVar(value=200)  # default value

        self.create_widgets()
        self.is_running = False
        self.quit_phrases = ["I quit", "stop", "exit"]
        self.queue = Queue()
        self.conversation_history = []

    def create_widgets(self):
        self.start_button = tk.Button(self, text="START", fg="green", command=self.start)
        self.start_button.pack(side="top")

        self.quit_button = tk.Button(self, text="QUIT", fg="red", command=self.quit)
        self.quit_button.pack(side="bottom")

        self.voice_selector = tk.OptionMenu(self, self.current_voice, *self.voices)
        self.voice_selector.pack()

        self.gpt3_radio = tk.Radiobutton(self, text="GPT-3.5 Turbo", variable=self.gpt_model, value="gpt-3.5-turbo")
        self.gpt3_radio.pack(side="left")
        
        self.gpt4_radio = tk.Radiobutton(self, text="GPT-4", variable=self.gpt_model, value="gpt-4")
        self.gpt4_radio.pack(side="left")

        self.max_tokens_slider = tk.Scale(self, from_=1, to=8000, orient="horizontal", variable=self.max_tokens, label="Max tokens")
        self.max_tokens_slider.pack()

        self.save_button = tk.Button(self, text="SAVE CONVERSATION", command=self.save_conversation)
        self.save_button.pack(side="bottom")

        self.conversation_display = tk.Text(self, state="disabled", width=250, height=50)
        self.conversation_display.pack()

    def save_conversation(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=(("JSON files", "*.json"), ("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")))
        if file_path:
            with open(file_path, "w") as f:
                if file_path.endswith(".json"):
                    json.dump(self.conversation_history, f)
                elif file_path.endswith(".csv"):
                    writer = csv.DictWriter(f, fieldnames=["role", "content"])
                    writer.writeheader()
                    for message in self.conversation_history:
                        writer.writerow(message)
                else:
                    for message in self.conversation_history:
                        f.write(f"{message['role']}: {message['content']}\n")

    def start(self):
        if not self.is_running:
            self.is_running = True
            threading.Thread(target=self.main).start()

    def quit(self):
        self.is_running = False

    def display_message(self, role, content):
        self.conversation_display.config(state="normal")
        self.conversation_display.insert("end", f"{role}: {content}\n")
        self.conversation_display.config(state="disabled")

    # Include the rest of your methods here...

# Your other methods go here...

if __name__ == "__main__":
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
