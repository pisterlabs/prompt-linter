import sys
import openai
import json

text_input = str(sys.argv[1])
identified_purpose = text_input
openai.api_key = "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"


def generate_code(prompt_received):
    completions = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt_received,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    message = completions.choices[0].text
    return message.strip()


prefix_file_list = "Generate a list of python code files related to ROS noetic implementation of given scenerio."
text_file_list = prefix_file_list + text_input
file_list_response = generate_code(text_file_list)
print(file_list_response)

filenames = file_list_response.split("\n")
filenames = [i.split(". ")[1] for i in filenames]
print(filenames)

import os

directory_name = "scripts"

if not os.path.exists(directory_name):
    os.makedirs(directory_name)


def get_code(filename, identified_purpose):
    generated_code = generate_code("generate a {} python code for {}.".format(filename, identified_purpose))
    return generated_code


import time

for filename in filenames:
    code = get_code(filename, identified_purpose)
    with open("scripts/" + filename, 'w') as f:
        f.write(code)
    time.sleep(1)

import os
import tkinter as tk


def display_files():
    # Get the directory path
    dir_path = "./scripts"
    # dir_path = tk.filedialog.askdirectory()

    # Get all files in the directory
    files = os.listdir(dir_path)

    # Create a Tkinter window
    root = tk.Tk()

    # Create a Listbox widget for file names
    listbox_names = tk.Listbox(root)
    listbox_names.pack(side=tk.LEFT)

    # Create a Text widget for file contents
    text_contents = tk.Text(root)
    text_contents.pack(side=tk.RIGHT)

    # Add each file to the Listbox
    for file in files:
        listbox_names.insert(tk.END, file)

    def show_contents(event):
        # Get the selected file
        selected_file = listbox_names.get(listbox_names.curselection())

        # Open the file and read its contents
        with open(os.path.join(dir_path, selected_file), "r") as f:
            contents = f.read()

        # Clear the Text widget and insert new contents
        text_contents.delete("1.0", tk.END)
        text_contents.insert(tk.END, contents)

    # Bind the Listbox to show_contents function
    listbox_names.bind("<<ListboxSelect>>", show_contents)

    # Start the Tkinter event loop
    root.mainloop()


display_files()
