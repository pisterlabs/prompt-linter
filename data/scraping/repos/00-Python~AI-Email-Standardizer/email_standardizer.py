import tkinter as tk
from tkinter import messagebox, scrolledtext, simpledialog
import openai
import os
import json

API_KEY_FILE = 'api_key.json'
parameters = "Rewrite the following copy in the style of a Tech Support operator, using short statements and bullet points"


def request(model, messages):
    response = openai.ChatCompletion.create(
        model=model, 
        temperature=1, 
        messages=messages
    )
    return response.choices[0].message["content"].strip()

def load_api_key():
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE, 'r') as file:
            key_info = json.load(file)
            openai.api_key = key_info.get("api_key")
    else:
        set_api_key()

def set_api_key():
    api_key = simpledialog.askstring("API Key", "Enter your API Key:")
    openai.api_key = api_key
    with open(API_KEY_FILE, 'w') as file:
        json.dump({"api_key": api_key}, file)
        
def load_parameters():
    global parameters
    parameter = simpledialog.askstring("Parameters", "Enter your desired parameters:", initialvalue=parameters)
    if parameter:
        parameters = parameter

def copy_to_clipboard():
    window.clipboard_clear()
    window.clipboard_append(result_text.get('1.0', tk.END))
    messagebox.showinfo("Info", "Text copied to clipboard")


def process_email():
    if not openai.api_key:
        messagebox.showerror("Error", "API Key is required.")
    else:
        data = email_text.get('1.0', tk.END).strip().replace('\n', ' ')
        messages = [
            {"role": "system", "content": f"{parameters}"},
            {"role": "user", "content": f"Email: {data}"}
        ]

        response = request('gpt-4', messages)
        result_text.insert(tk.INSERT, response)
        
def load_parameters():
    global parameters
    top = tk.Toplevel()
    top.title("Parameters")
    top.grid_columnconfigure(0, weight=1)

    parameter_label = tk.Label(top, text="Enter your desired parameters:")
    parameter_label.grid(row=0, column=0, sticky='w')

    parameter_entry = tk.Entry(top, width=50)   
    parameter_entry.insert(0, parameters)
    parameter_entry.grid(row=1, column=0, sticky='ew')

    ok_button = tk.Button(top, text="OK", command=lambda: _update_parameter_and_destroy(top, parameter_entry))
    ok_button.grid(row=2, column=0)
    
    
def _update_parameter_and_destroy(top, entry):
    global parameters
    parameters = entry.get()
    top.destroy()

load_api_key()

# Create a new tkinter window
window = tk.Tk()
window.grid_columnconfigure(0, weight=1)
window.grid_rowconfigure(0, weight=1)


# Create a menu for the parameters and exit
menu = tk.Menu(window)

# Creates a button in the menu to load parameters
param_menu = tk.Menu(menu, tearoff=0)
param_menu.add_command(label="Load Parameters", command=load_parameters)
param_menu.grid_columnconfigure(0, weight=1)  # make the parameter box responsive

menu.add_cascade(label="Menu", menu=param_menu)
menu.add_command(label="Exit", command=window.quit)

window.config(menu=menu)

# Create a new textbox for email input
email_text = scrolledtext.ScrolledText(window, width=50, height=20)
email_text.grid(row=0, column=0, columnspan=2, sticky='nsew', pady=(0, 5))  # give the box a small padding to the bottom

# Create a new button that will process the email when clicked
process_button = tk.Button(window, text="Process Email", command=process_email)
process_button.grid(row=1, column=0, columnspan=2)

# Create a textbox to display the result
result_text = scrolledtext.ScrolledText(window, width=50, height=20)
result_text.grid(row=2, column=0, columnspan=2, sticky='nsew', pady=(5, 0))  # give the box a small padding to the top

# Create a new button that will copy the result to clipboard when clicked
copy_button = tk.Button(window, text="Copy to Clipboard", command=copy_to_clipboard)
copy_button.grid(row=3, column=0, columnspan=2)

window.mainloop()
