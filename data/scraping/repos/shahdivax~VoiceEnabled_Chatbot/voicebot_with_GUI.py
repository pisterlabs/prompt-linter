import threading
import tkinter as tk
from tkinter import filedialog

import openai
import speech_recognition as sr
import pyttsx3


class Voice_ChatBot:
    def __init__(self):
        self.window = tk.Tk()
        self.window.geometry("700x550")
        self.window.title("Voice ChatBot")
        self.window.resizable(False, False)
        # Create a label and entry box for the OpenAPI key
        self.api_key_label = tk.Label(self.window, text="OpenAI API key:")
        self.api_key_entry = tk.Entry(self.window, width=40)
        self.api_key_label.pack()
        self.api_key_entry.pack()
        self.button1 = tk.Button(self.window, text="Submit", command=self.getkey)
        self.button1.pack(padx=10, pady=10)
        self.label = tk.Label(self.window, text="")
        self.label.pack(padx=10, pady=10)

        # Create a textbox for displaying text
        self.textbox = tk.Text(self.window, width=100, height=20, bg='#F6F6F6', fg='black', wrap="word")
        self.textbox.tag_config('user', foreground='green')
        self.textbox.tag_config('bot', foreground='blue')
        self.textbox.tag_config('system', foreground='red')
        self.textbox.yview_pickplace("end")
        self.textbox.pack(padx=10)

        # Create buttons for executing actions

        self.button2 = tk.Button(self.window, text="Speak", command=self.start_thread, background="gray")
        self.button2.pack(padx=10, pady=10)

        self.button3 = tk.Button(self.window, text="Save To Text File", command=self.save_file, background="skyblue")
        self.button3.pack(padx=20, pady=10)

        self.window.mainloop()

    # get api_key
    def getkey(self):
        global openaikey
        openaikey = self.api_key_entry.get()
        openai.api_key = openaikey
        # self.textbox.insert(tk.END, "Click Speak to speak: \n", 'system')
        self.label.config(text="Click Speak Button once to speak", foreground="red")
        self.api_key_entry.delete(0, 'end')

    # define function to capture user's voice input
    def get_voice_input(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio = r.listen(source)
        try:
            text_input = r.recognize_google(audio)
            self.textbox.insert(tk.END, "You: " + text_input + "\n", 'user')
            response = self.generate_response(text_input)
            self.generate_speech_response(response)
        except sr.UnknownValueError:
            self.textbox.insert(tk.END, "Could not recognize input. \n", 'system')
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return ""

    # define function to generate text response using OpenAI API
    def generate_response(self, input_text):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=input_text,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        output = response.choices[0].text.strip()
        self.textbox.insert(tk.END, "Chatbot: " + output + "\n", 'bot')
        return output

    # define function to convert text response to speech
    def generate_speech_response(self, text_response):
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id)
        engine.say(text_response)
        engine.runAndWait()
        self.textbox.insert(tk.END, "Speak: \n", 'system')

    def start(self):
        self.label.config(text="")
        while True:
            user_input = self.get_voice_input()
            if user_input:
                response = self.generate_response(user_input)
                self.generate_speech_response(response)

    def start_thread(self):
        self.textbox.insert(tk.END, "Speak: \n", 'system')
        threading.Thread(target=self.start).start()

    def save_file(self):
        # Get the text from the Text widget
        text = self.textbox.get("1.0", "end-1c")

        # Open a file dialog to choose the save location
        file_path = filedialog.asksaveasfilename(defaultextension=".txt")

        # Check if a file was chosen and write the text to it
        if file_path:
            with open(file_path, "w") as f:
                f.write(text)
            self.label.config(text="File saved successfully!", foreground="darkblue")
        else:
            self.label.config(text="File not saved.!!!", foreground="red")


gui = Voice_ChatBot()
