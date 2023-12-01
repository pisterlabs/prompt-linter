import datetime
import configparser
from base64 import b64decode
import webbrowser
import openai
from openai.error import InvalidRequestError
import tkinter as tk
from tkinter import messagebox
from tkinter.font import Font

def generate_image(prompt, num_image=1, size='512x512', output_format='url'):
    try:
        images = []
        response = openai.Image.create(
            prompt=prompt,
            n=num_image,
            size=size,
            response_format=output_format
        )
        if output_format == 'url':
            for image in response['data']:
                images.append(image.url)
        elif output_format == 'b64_json':
            for image in response['data']:
                images.append(image.b64_json)
        return {'created': datetime.datetime.fromtimestamp(response['created']), 'images': images}
    except InvalidRequestError as e:
        print(e)

def generate_images():
    prompt = text_entry.get("1.0", "end-1c")
    num_image = int(num_images_entry.get())
    size = selected_size.get()
    if size not in SIZES:
        messagebox.showerror("Invalid Input", "Please select a valid resolution.")
        return

    response = generate_image(prompt, num_image, size)
    images = response['images']
    for image in images:
        webbrowser.open(image)

def select_resolution(size):
    selected_size.set(size)
    for button in resolution_buttons:
        if button['text'] == size:
            button.config(bg=SELECTED_COLOR)
        else:
            button.config(bg=PRIMARY_COLOR)

config = configparser.ConfigParser()
config.read('credential.ini')

openai.api_key = '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'

SIZES = ('1024x1024', '512x512', '256x256')

# ChatGPT Theme Colors
PRIMARY_COLOR = "#4E8EE9"  # Blue
SECONDARY_COLOR = "#FFA500"  # Orange
BACKGROUND_COLOR = "#252525"  # Dark Gray
TEXT_COLOR = "#FFFFFF"  # White
SELECTED_COLOR = "#FFD700"  # Gold

# Create the main window
root = tk.Tk()
root.title("Image Generation")
root.geometry("400x300")
root.configure(bg=BACKGROUND_COLOR)

# Define custom fonts
title_font = Font(family="Helvetica", size=24, weight="bold")
label_font = Font(family="Helvetica", size=16)
button_font = Font(family="Helvetica", size=14)

# Title Label
title_label = tk.Label(root, text="Image Generation", bg=BACKGROUND_COLOR, fg=PRIMARY_COLOR, font=title_font)
title_label.pack(pady=10)

# Text Entry
text_label = tk.Label(root, text="Enter the text:", bg=BACKGROUND_COLOR, fg=TEXT_COLOR, font=label_font)
text_label.pack()
text_entry = tk.Text(root, height=5, font=label_font, bg=PRIMARY_COLOR, fg=TEXT_COLOR)
text_entry.pack()

# Number of Images Entry
num_images_label = tk.Label(root, text="Number of Images:", bg=BACKGROUND_COLOR, fg=TEXT_COLOR, font=label_font)
num_images_label.pack()
num_images_entry = tk.Entry(root, font=label_font, bg=PRIMARY_COLOR, fg=TEXT_COLOR)
num_images_entry.pack()

# Resolution Selection
sizes_label = tk.Label(root, text="Resolution:", bg=BACKGROUND_COLOR, fg=TEXT_COLOR, font=label_font)
sizes_label.pack()

resolution_frame = tk.Frame(root, bg=BACKGROUND_COLOR)
resolution_frame.pack()

selected_size = tk.StringVar(root, SIZES[1])  # Set default resolution to '512x512'

resolution_buttons = []
for size in SIZES:
    size_button = tk.Button(resolution_frame, text=size, command=lambda size=size: select_resolution(size),
                            bg=PRIMARY_COLOR, fg=TEXT_COLOR, activebackground=PRIMARY_COLOR,
                            activeforeground=TEXT_COLOR, font=label_font)
    size_button.pack(side="left", padx=5, pady=5)
    resolution_buttons.append(size_button)

# Generate Button
generate_button = tk.Button(root, text="Generate", command=generate_images, bg=PRIMARY_COLOR, fg=TEXT_COLOR,
                            activebackground=SECONDARY_COLOR, activeforeground=TEXT_COLOR, font=button_font)
generate_button.pack(pady=10)

# Start the main event loop
root.mainloop()
