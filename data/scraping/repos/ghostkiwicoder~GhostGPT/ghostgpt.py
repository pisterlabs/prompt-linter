import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from tkinter import filedialog, ttk
import speech_recognition as sr
import azure.cognitiveservices.speech as speechsdk
import openai
import os
import threading
import webbrowser
import xml.etree.ElementTree as ET
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, AudioConfig
from azure.core.credentials import AzureKeyCredential
import sounddevice




def open_github_link(event):
    webbrowser.open("https://github.com/ghostkiwicoder")

def create_and_open_api_key_file():
    api_key_file = "apikey.xml"
    if not os.path.isfile(api_key_file):
        root = ET.Element("config")
        ET.SubElement(root, "azure_api_key").text = "your-azure-api-key"
        ET.SubElement(root, "azure_region").text = "your-azure-region"
        ET.SubElement(root, "openai_api_key").text = "your-openai-api-key"
        tree = ET.ElementTree(root)
        tree.write(api_key_file)
    os.startfile(api_key_file)
    check_api_key_file()

def check_api_key_file():
    api_key_file = "apikey.xml"
    if not os.path.isfile(api_key_file):
        api_key_status.config(text="Please load your API key", fg="red")
    else:
        tree = ET.parse(api_key_file)
        root = tree.getroot()

        azure_api_key = root.find("azure_api_key").text.strip() if root.find("azure_api_key") is not None else ""
        azure_region = root.find("azure_region").text.strip() if root.find("azure_region") is not None else ""
        openai_api_key = root.find("openai_api_key").text.strip() if root.find("openai_api_key") is not None else ""

        if (azure_api_key and azure_api_key != "your-azure-api-key" and
                azure_region and azure_region != "your-azure-region" and
                openai_api_key and openai_api_key != "your-openai-api-key"):
            api_key_status.config(text="API Key Ready", fg="green")
            load_voices(azure_api_key, azure_region)
            openai.api_key = openai_api_key
            return azure_region, azure_api_key
        else:
            api_key_status.config(text="Please load your API key", fg="red")
            return None, None


def export_chat():
    file_path = filedialog.asksaveasfilename(defaultextension=".txt")
    if file_path:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(chat_log.get(1.0, "end"))

def talk_to_gpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an adept assistant, programmed to deliver the most accurate and relevant responses to any user prompt. Limit responces to cut off after 150 words maximum"},
            {"role": "user", "content": prompt}
        ],
    )
    return response['choices'][0]['message']['content']

def send_message():
    message = entry.get()
    if not message:
        return
    chat_log.config(state="normal")
    chat_log.insert("end", f"User: {message}\n")
    chat_log.tag_configure("user", foreground="white")
    chat_log.tag_add("user", chat_log.index("end - 2 lines linestart"), chat_log.index("end - 2 lines lineend"))
    chat_log.see("end")
    entry.delete(0, "end")
    response    = talk_to_gpt(message)
    chat_log.insert("end", f"GhostGPT: {response}\n")
    chat_log.tag_configure("gpt", foreground="cyan")
    chat_log.tag_add("gpt", chat_log.index("end - 2 lines linestart"), chat_log.index("end - 2 lines lineend"))
    chat_log.see("end")
    chat_log.config(state="disabled")
    threading.Thread(target=speak, args=(response,)).start()

# global flag for stopping speech
stop_speech = False

def speak(text):
    global stop_speech
    stop_speech = False

    tree = ET.parse("apikey.xml")
    root = tree.getroot()

    azure_api_key = root.find("azure_api_key").text.strip()
    azure_region = root.find("azure_region").text.strip()

    speech_config = SpeechConfig(subscription=azure_api_key, region=azure_region)
    speech_config.speech_synthesis_voice_name = voice_combobox.get()
    speech_config.speech_synthesis_rate = speed_slider.get()

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    sentences = text.split('.')
    chunk = ""

    for sentence in sentences:
        if len(chunk) + len(sentence) < 1024:
            chunk += sentence + '.'
        else:
            synthesizer.speak_text_async(chunk).get()
            chunk = sentence + '.'

        if stop_speech:
            break

    if chunk:
        synthesizer.speak_text_async(chunk).get()

def stop_talking():
    global stop_speech
    stop_speech = True

def listen_to_user():
    def threaded_listen():
        if start_button["text"] == "Start Listening":
            start_button.config(text="Stop Listening")
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                audio = recognizer.listen(source, phrase_time_limit=5, timeout=5)
            try:
                user_text = recognizer.recognize_google(audio)
                entry.delete(0, "end")
                entry.insert(0, user_text)
                send_message()
            except sr.UnknownValueError:
                entry.delete(0, "end")
                entry.insert(0, "Voice not recognized.")
            start_button.config(text="Start Listening")
        else:
            start_button.config(text="Start Listening")

    threading.Thread(target=threaded_listen).start()

def clear_chat():
    chat_log.config(state="normal")
    chat_log.delete(1.0, "end")
    chat_log.config(state="disabled")

def load_voices(api_key, region):
    speech_config = SpeechConfig(subscription=api_key, region=region)
    synthesizer = SpeechSynthesizer(speech_config=speech_config)

    voices_result = synthesizer.get_voices_async().get()
    voice_names = [voice.name for voice in voices_result.voices]

    voice_combobox["values"] = voice_names
    default_voice_index = voice_names.index("Microsoft Server Speech Text to Speech Voice (en-US, SaraNeural)")
    voice_combobox.current(default_voice_index)  # set the default voice to Sara

    # populate output device dropdown list
    output_device_names = []
    for device in sounddevice.query_devices():
        if device["max_output_channels"] > 0:
            output_device_names.append(device["name"])

    output_combobox["values"] = output_device_names
    if len(output_device_names) > 0:
        output_combobox.current(0)  # set the default output device

root = tk.Tk()
root.title("GhostGPT")
root.configure(bg="black")

chat_log = ScrolledText(root, wrap="word", width=50, height=20, bg="black", fg="white", state="disabled")
chat_log.grid(row=0, column=0, columnspan=3, padx=5, pady=5)

entry = tk.Entry(root, width=40, bg="black", fg="white")
entry.grid(row=1, column=0, padx=5, pady=5)

send_button = tk.Button(root, text="Send", command=send_message, bg="black", fg="white", width=10)
send_button.grid(row=1, column=1, padx=5, pady=5)

start_button = tk.Button(root, text="Start Listening", command=listen_to_user, bg="black", fg="white", width=10)
start_button.grid(row=2, column=0, padx=5, pady=5)

clear_button = tk.Button(root, text="Clear", command=clear_chat, bg="black", fg="white", width=10)
clear_button.grid(row=2, column=1, padx=5, pady=5)

export_button = tk.Button(root, text="Export Chat", command=export_chat, bg="black", fg="white", width=10)
export_button.grid(row=3, column=0, padx=5, pady=5)

api_key_button = tk.Button(root, text="API Key", command=create_and_open_api_key_file, bg="black", fg="white", width=10)
api_key_button.grid(row=3, column=1, padx=5, pady=5)

api_key_status = tk.Label(root, text="Loading API Key...", fg="red", bg="black")
api_key_status.grid(row=4, column=0, columnspan=3, padx=5, pady=5)

voice_label = tk.Label(root, text="Voices", fg="white", bg="black")
voice_label.grid(row=5, column=0, padx=5, pady=5)

voice_combobox = ttk.Combobox(root, state="readonly", width=70)
voice_combobox.grid(row=6, column=0, padx=5, pady=5)

output_label = tk.Label(root, text="Output", fg="white", bg="black")
output_label.grid(row=7, column=0, padx=5, pady=5)

output_combobox = ttk.Combobox(root, state="readonly", width=70)
output_combobox.grid(row=8, column=0, padx=5, pady=5)

speed_label = tk.Label(root, text="Speaking Speed", fg="white", bg="black")
speed_label.grid(row=9, column=0, padx=5, pady=5)

speed_slider = tk.Scale(root, from_=0, to=10, length=400, orient="horizontal", sliderlength=10, resolution=0.1)
speed_slider.set(1)
speed_slider.grid(row=10, column=0, padx=5, pady=5)

version_label = tk.Label(root, text="GhostGPT by Ghost v.1.0.0", fg="red", bg="black", font=("Helvetica", 13, "italic"))
version_label.grid(row=11, column=0, padx=5, pady=0)

github_link = tk.Label(root, text="github.com/ghostkiwicoder", fg="cyan", cursor="hand2", bg="black")
github_link.grid(row=12, column=0, padx=5, pady=5)
github_link.bind("<Button-1>", open_github_link)

stop_button = tk.Button(root, text="Stop Speaking", command=stop_talking, bg="black", fg="white", width=10)
stop_button.grid(row=10, column=1, padx=5, pady=5)


# initializing Azure TTS
azure_region, azure_api_key = check_api_key_file()

if azure_region and azure_api_key:
    azure_tts_client = speechsdk.SpeechConfig(subscription=azure_api_key, region=azure_region)
    api_key_status.config(text="API Key Loaded", fg="green")
    load_voices(azure_api_key, azure_region)
else:
    api_key_status.config(text="No API Key Found", fg="red")

root.mainloop()