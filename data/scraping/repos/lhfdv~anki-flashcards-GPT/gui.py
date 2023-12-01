import os
import subprocess
import threading
import openai

from dotenv import load_dotenv

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
from tkinter import simpledialog
from flashcard_generator import generate_flashcards

load_dotenv()

# Variables
default_content = "Example: word1, word2, word3..."
output_folder = "output"
api_key = os.environ.get("API_KEY", "")
open_folder_after_generation = True
gpt_model = "gpt-3.5-turbo" 

def clear_default_content(event):
    current_content = entry.get("1.0", tk.END).strip()
    if current_content == default_content:
        entry.delete("1.0", tk.END)

def button_click():
    input_text = entry.get("1.0", tk.END).strip()
    language = language_combobox.get()  
    input_language = input_language_combobox.get() 

    if not input_text:
        tk.messagebox.showerror("Error", "No input")
        return

    window.button.config(state=tk.DISABLED)

    thread = threading.Thread(
        target=generate_flashcards,
        args=(input_text, language, input_language, window.loading_label, window.button, window.progress_bar, output_folder, gpt_model),
    )
    thread.start()

def about():
    about_text = "Flashcard Generator\nVersion 0.1.1\nGitHub: https://github.com/lhfdv"
    messagebox.showinfo("About", about_text)

def open_output_folder():
    folder_path = os.path.abspath(output_folder)
    subprocess.Popen(["explorer", folder_path])

def change_output_folder():
    global output_folder
    folder_path = filedialog.askdirectory(initialdir=os.getcwd(), title="Select Output Folder")
    if folder_path:
        output_folder = folder_path
        messagebox.showinfo("Output Folder", f"Selected Output Folder: {output_folder}")

def toggle_open_folder_option():
    global open_folder_after_generation
    open_folder_after_generation = not open_folder_after_generation

def change_api_key():
    global api_key
    api_key = simpledialog.askstring("API Key", "Enter your OpenAI API Key:")
    if api_key:
        os.environ["API_KEY"] = api_key
        openai.api_key = api_key

def change_gpt_model():
    global gpt_model
    gpt_model = simpledialog.askstring("GPT Model", "Enter the GPT model name (e.g., gpt-4.0):")
    if gpt_model:
        messagebox.showinfo("GPT Model", f"Selected GPT Model: {gpt_model}")

def close():
    window.destroy()

openai.api_key = api_key

# Main window settings
window = tk.Tk()
window.title("Flashcard Generator")

# Menu
menu = tk.Menu(window)
window.config(menu=menu)

# "File"
file_menu = tk.Menu(menu, tearoff=0)
menu.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Output Folder", command=open_output_folder)

# "Options"
options_menu = tk.Menu(menu, tearoff=0)
menu.add_cascade(label="Options", menu=options_menu)
options_menu.add_command(label="Change Output Folder", command=change_output_folder)
options_menu.add_command(label="Open Folder After Generation", command=toggle_open_folder_option)
options_menu.add_command(label="Change API Key", command=change_api_key)
options_menu.add_command(label="Change GPT Model", command=change_gpt_model)

# "About"
about_menu = tk.Menu(menu, tearoff=0)
about_menu.add_command(label="About", command=about)
menu.add_cascade(label="Help", menu=about_menu)

# "Close"
menu.add_command(label="Close", command=close)

# Window fixed size
window_width = 555
window_height = 285
window.geometry(f"{window_width}x{window_height}")

# Label for the input
label = tk.Label(window, text="Enter words to generate cards:")
label.grid(row=0, column=0, columnspan=2, pady=10)

# Input field for words
entry = tk.Text(window, height=5)
entry.grid(row=1, column=0, columnspan=2, pady=5)
entry.insert("1.0", default_content)
entry.bind("<FocusIn>", clear_default_content)

# Label for the language selection (Input)
input_language_label = tk.Label(window, text="Select input language:")
input_language_label.grid(row=2, column=0, pady=5)

language_combobox = ttk.Combobox(window, values=["Spanish", "French", "Italian", "Japanese"])
language_combobox.current(0)
language_combobox.grid(row=3, column=0, pady=5)

# Label for the language selection (Output)
language_label = tk.Label(window, text="Select output language:")
language_label.grid(row=2, column=1, pady=5)

input_language_combobox = ttk.Combobox(window, values=["Brazilian Portuguese", "English"])
input_language_combobox.current(0)
input_language_combobox.grid(row=3, column=1, pady=5)

# Label for the loading message
loading_label = tk.Label(window, text="")
window.loading_label = loading_label
loading_label.grid(row=4, columnspan=2)

# Progress bar - Incomplete
progress_bar = ttk.Progressbar(window, mode="determinate")
window.progress_bar = progress_bar
progress_bar.grid(row=5, columnspan=2, pady=5)

# Generate button
window.button = tk.Button(window, text="Generate", command=button_click)
window.button.grid(row=6, columnspan=2, pady=5)

window.mainloop()
