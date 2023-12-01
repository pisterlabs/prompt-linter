import tkinter as tk
import threading
import openai
import speech_recognition as sr
import pyttsx3

key = 'openaikey'
personality = "personality.txt"
usewhisper = True

openai.api_key = key
with open(personality, "r") as file:
    mode = file.read()
messages = [{"role": "system", "content": f"{mode}"}]

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

r = sr.Recognizer()
mic = sr.Microphone(device_index=0)
r.dynamic_energy_threshold = False
r.energy_threshold = 400

listening_flag = threading.Event()
listening_flag.set()

stop_phrases =["knox gpt","knox gpt","see you later gpt","see you later GPT","signout gpt","signout GPT","bye bye gpt","bye bye GPT", "signout bot","logout bot","stop bot","end bot"]

root = tk.Tk()
root.title("Avvai AI integrated Voice-Bot")
root.geometry("800x600")
root.configure(bg="#2c3e50")

title_label = tk.Label(root, text="Avvai587 AI integrated Voice-Bot. Ask me anything! ;)", font=("Helvetica", 20, "bold"), bg="#2c3e50", fg="white")
title_label.pack(pady=20)

conversation_display = tk.Text(root, wrap=tk.WORD, state=tk.DISABLED, font=("Helvetica", 12), height=20, bg="#34495e", fg="white")
conversation_display.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

def handle_user_input():
    while listening_flag.is_set():
        with mic as source:
            update_display("Listening... 🔊\n\n")
            print("\nListening...")
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source)

            try:
                user_input = r.recognize_google(audio)
                print(f"User: {user_input}")

                if any(phrase in user_input.lower() for phrase in stop_phrases):
                    print("Stopping conversation...")
                    stop_listening()
                    update_display("Listening has been stopped... 🛑\nPlease click 'Start' to resume the conversation.\n\n")
                    break

                messages.append({"role": "user", "content": user_input})

                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0301",
                    messages=messages,
                    temperature=0.8
                )

                response = completion.choices[0].message.content
                messages.append({"role": "assistant", "content": response})

                update_display(f"User: {user_input} 😄\nAssistant: {response} 🤖\n\n")

                engine.say(response)
                engine.runAndWait()

                conversation_display.see(tk.END)
            except sr.UnknownValueError:
                print("Sorry, I didn't catch that.")

        root.after(100)

def start_listening():
    global speech_thread
    listening_flag.set()
    speech_thread = threading.Thread(target=handle_user_input)
    speech_thread.daemon = True
    speech_thread.start()

def stop_listening():
    global speech_thread
    listening_flag.clear()
    update_display("Listening has been stopped... 🛑\nPlease click 'Start' to resume the conversation.\n\n")

start_button = tk.Button(root, text="Start", font=("Helvetica", 14), bg="#27ae60", fg="white", command=start_listening)
start_button.pack(pady=10)

stop_button = tk.Button(root, text="Stop", font=("Helvetica", 14), bg="#c0392b", fg="white", command=stop_listening)
stop_button.pack(pady=10)

def update_display(message):
    conversation_display.config(state=tk.NORMAL)
    conversation_display.insert(tk.END, message)
    conversation_display.config(state=tk.DISABLED)
    conversation_display.see(tk.END)

root.mainloop()
