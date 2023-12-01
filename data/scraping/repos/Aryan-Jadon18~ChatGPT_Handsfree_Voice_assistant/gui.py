import tkinter as tk
import threading
import openai
import speech_recognition as sr
import pyttsx3

openai.api_key = 'YOUR_API_KEY_HERE' #REPLACE WITH YOUR OPENAI API KEY, KEEP THE QUOTES

messages = [
    {"role": "system", "content": "You are a kind helpful assistant."},
]

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def transcribe_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        output_text.config(state=tk.NORMAL)
        output_text.insert(tk.END, "Listening...\n")
        output_text.config(state=tk.DISABLED)
        output_text.see(tk.END)
        output_text.update()

        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("Unable to recognize speech")
    except sr.RequestError as e:
        print(f"Error occurred during speech recognition: {e}")

def start_listening():
    voice_input_text = transcribe_speech()
    output_text.config(state=tk.NORMAL)
    output_text.insert(tk.END, f"User (Voice): {voice_input_text}\n")
    output_text.config(state=tk.DISABLED)
    output_text.see(tk.END)
    output_text.update()

    messages.append({"role": "user", "content": voice_input_text})

    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    assistant_reply = chat.choices[0].message.content
    output_text.config(state=tk.NORMAL)
    output_text.insert(tk.END, f"ChatGPT: {assistant_reply}\n")
    output_text.config(state=tk.DISABLED)
    output_text.see(tk.END)
    output_text.update()

    messages.append({"role": "assistant", "content": assistant_reply})
    speak(assistant_reply)

def send_message():
    prompt_input = input_text.get()
    if prompt_input == "no":
        root.destroy()
    else:
        messages.append({"role": "user", "content": prompt_input})

        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        assistant_reply = chat.choices[0].message.content
        output_text.config(state=tk.NORMAL)
        output_text.insert(tk.END, f"User: {prompt_input}\n")
        output_text.insert(tk.END, f"ChatGPT: {assistant_reply}\n")
        output_text.config(state=tk.DISABLED)
        output_text.see(tk.END)
        output_text.update()

        messages.append({"role": "assistant", "content": assistant_reply})
        speak(assistant_reply)

    input_text.delete(0, tk.END)

def voice_input():
    threading.Thread(target=start_listening).start()

root = tk.Tk()
root.title("AI Assistant")

input_frame = tk.Frame(root)
input_frame.pack(pady=10)

input_text = tk.Entry(input_frame, width=50)
input_text.pack(side=tk.LEFT)

send_button = tk.Button(input_frame, text="Send", command=send_message)
send_button.pack(side=tk.LEFT, padx=10)

voice_button = tk.Button(root, text="Voice Input", command=voice_input)
voice_button.pack(pady=10)

output_text = tk.Text(root, height=10, width=50)
output_text.pack(pady=10)
output_text.config(state=tk.DISABLED)

root.mainloop()
