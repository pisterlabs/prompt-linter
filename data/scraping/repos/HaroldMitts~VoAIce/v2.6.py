import json
import os
import subprocess
import tempfile
import time
import azure.cognitiveservices.speech as speechsdk
import openai
import tkinter as tk
from tkinter import filedialog
import csv
import threading
from queue import Queue

def clear_screen():
    os.system("cls") if os.name == "nt" else os.system("clear")

if __name__ == "__main__":
    clear_screen()
    
def load_api_keys(file_path):
    with open(file_path, "r") as f:
        keys = json.load(f)
    return keys

def transcribe_audio(speech_config):
    audio_config = speechsdk.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    result = speech_recognizer.recognize_once_async().get()
    return result.text.strip()

def generate_response(input_text, conversation_history):
    messages = "Think step by step before answering any question. \
            reflect on the question and all potential answers. Examine the flaw and faulty \
            logic of each answer option and eliminate the ones that are incorrect. \
            If you are unsure of the answer, make an educated guess. \
            If you are still unsure, eliminate the answers that you know are incorrect and then guess \
            from the remaining answers. \
            If any answer provided is comprised of an educated guess, make sure to note that in your answer."

    for conversation in conversation_history:
        role = conversation['role']
        content = conversation['content']
        messages += f'\n{role}: {content}'

    messages += f'\nUser: {input_text}'

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=messages,
        max_tokens=self.max_tokens.get(),
        temperature=self.temperature.get(),
        n=1,
        stop=None
    )

    assistant_response = response['choices'][0]['text']
    return assistant_response

def synthesize_and_save_speech(speech_config, response_text, file_path):
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
    result = speech_synthesizer.speak_text_async(response_text).get()

    with open(file_path, "wb") as f:
        f.write(result.audio_data)

def play_audio(audio_file_path):
    subprocess.call(["ffplay", "-nodisp", "-autoexit", audio_file_path], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()

        with open("voices.json", "r") as f:
            voice_data = json.load(f)

        self.voice_map = {voice['DisplayName']: voice['ShortName'] for voice in voice_data}
        self.voice_map = dict(sorted(self.voice_map.items(), key=lambda item: item[0]))
        self.current_voice = tk.StringVar(self.master)
        self.current_voice.set(list(self.voice_map.keys())[0]) 

        self.model_choice = tk.StringVar(self.master)
        self.model_choice.set("gpt-4")

        self.max_tokens = tk.IntVar(self.master)
        self.max_tokens.set(200) # Initial value

        self.temperature = tk.DoubleVar(self.master)
        self.temperature.set(1.0) # Initial value

        self.create_widgets()
        self.is_running = False
        self.quit_phrases = ["I quit", "stop", "exit"]
        self.queue = Queue()
        self.conversation_history = []

    def create_widgets(self):
        self.start_button = tk.Button(self)
        self.start_button["text"] = "Start"
        self.start_button["command"] = self.start
        self.start_button.pack(side="top")

        self.conversation_display = tk.Text(self, state="disabled", width=100, height=50)
        self.conversation_display.pack(side="top")

        self.voice_selector = tk.OptionMenu(self, self.current_voice, *self.voice_map.keys())
        self.voice_selector.pack(side="top")

        self.model_radio_button_gpt4 = tk.Radiobutton(self, text="GPT-4", variable=self.model_choice, value="gpt-4")
        self.model_radio_button_gpt4.pack(side="top")

        self.model_radio_button_gpt35 = tk.Radiobutton(self, text="GPT-3.5", variable=self.model_choice, value="text-davinci-003")
        self.model_radio_button_gpt35.pack(side="top")

        self.max_tokens_slider = tk.Scale(self, from_=1, to=8000, orient="horizontal", label="Max Tokens", variable=self.max_tokens)
        self.max_tokens_slider.pack(side="top")

        self.temperature_slider = tk.Scale(self, from_=0.0, to=1.5, resolution=0.01, orient="horizontal", label="Temperature", variable=self.temperature)
        self.temperature_slider.pack(side="top")

        self.save_button = tk.Button(self)
        self.save_button["text"] = "Save"
        self.save_button["command"] = self.save_conversation
        self.save_button.pack(side="top")

    def update_display(self):
        while self.is_running:
            if not self.queue.empty():
                text = self.queue.get()
                self.conversation_display.configure(state="normal")
                self.conversation_display.insert("end", text)
                self.conversation_display.configure(state="disabled")
                self.conversation_display.see("end")

    def start(self):
        self.is_running = True
        threading.Thread(target=self.main).start()
        threading.Thread(target=self.update_display).start()

    def stop(self):
        self.is_running = False

    def save_conversation(self):
        filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Text Files", "*.txt"), ("CSV Files", "*.csv"), ("JSON Files", "*.json")])
        if not filename:
            return
        if filename.endswith(".json"):
            with open(filename, "w") as f:
                json.dump(self.conversation_history, f)
        elif filename.endswith(".csv"):
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(self.conversation_history)
        else:
            with open(filename, "w") as f:
                for item in self.conversation_history:
                    f.write("%s\n" % item)

    def main(self):
        keys_file_path = r"C:\\keys\\keys.json"
        keys = load_api_keys(keys_file_path)
        azure_api_key = keys["azure_api_key"]
        azure_region = keys["azure_region"]
        openai_api_key = keys["openai_api_key"]
        speech_config = speechsdk.SpeechConfig(subscription=azure_api_key, region=azure_region)
        speech_config.speech_synthesis_voice_name = self.voice_map[self.current_voice.get()]
        openai.api_key = openai_api_key

        while self.is_running:
            self.queue.put("Listening...\n")
            input_text = transcribe_audio(speech_config)
            self.queue.put(f"Input: {input_text}\n")

            if any(phrase.lower() in input_text.lower() for phrase in self.quit_phrases):
                break

            response_text = generate_response(input_text, self.conversation_history)
            self.queue.put(f"Response: {response_text}\n")

            self.conversation_history.append({"role": "user", "content": input_text})
            self.conversation_history.append({"role": "assistant", "content": response_text})

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                audio_file_path = f.name

            try:
                synthesize_and_save_speech(speech_config, response_text, audio_file_path)
            except Exception as e:
                self.queue.put(f"Error: Failed to synthesize speech - {e}\n")

            try:
                play_audio(audio_file_path)
            except Exception as e:
                self.queue.put(f"Error: Failed to play WAV file - {e}\n")

            os.remove(audio_file_path)

        self.queue.put("End of conversation.\n")
        self.stop()

root = tk.Tk()
app = Application(master=root)
app.mainloop()
