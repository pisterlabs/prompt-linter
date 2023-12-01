#JMR GUI Art Maker updated  GUI July 3 2023 runs on GPUs on MAC!!
#July 8 added stable-diffusion-xl-base-0.9!
import os
import torch
import tqdm
import re
from PIL import Image
from datetime import datetime
#from diffusers import StableDiffusionPipeline
from diffusers import DiffusionPipeline # may need for xl-base
import openai
from io import BytesIO
from base64 import b64decode
import tkinter as tk
from tkinter import Tk
from tkinter import filedialog, Scale
from tkinter import messagebox

#path_local = "/Users/jonathanrothberg/"

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
    print (mps_device)
else:
    print ("MPS device not found.")

# Create the main window
root = tk.Tk()
root.title("JMR's Text to Drawing")
root.geometry("800x450")

default_path = "/Users/jonathanrothberg"

if not os.path.exists(default_path):
    messagebox.showinfo("Information","Select Directory for Stable diffusion models") #crashes my mac if no root window.
    path_local = filedialog.askdirectory(title="Select Directory for Stable diffusion models")
else:
    path_local = default_path
print (path_local)

art_directory = "/Users/jonathanrothberg/AIArt"

# Check if the default path with the model exists
if not os.path.exists(os.path.join (default_path, art_directory)):
    messagebox.showinfo("Information","Select Directory to save your Art")
    image_dir = filedialog.askdirectory(title="Select Directory to save your Art")
    
else:
    image_dir = art_directory
print (image_dir)


API_ENDPOINT = "https://api.openai.com/v1/chat/completions"

API_KEY = input ("enter your API AI key: ")

openai.api_key = API_KEY

model_ids = [
    "stable-diffusion-v1-4", 
    "stable-diffusion-v1-5", 
    "stable-diffusion-2",
    "stable-diffusion-2-1",
    "stable-diffusion-xl-base-1.0",
    "DaleE" 
]

# Function to sanitize filenames
def sanitize_filename(filename):
    return re.sub(r'[^a-zA-Z0-9\-\_\.]', '_', filename)

# Function to generate images
def generate_images():
    # Get values from the GUI
    model_selection = [model_id for model_id, var in zip(model_ids, model_ids_vars) if var.get() == 1]
    print ("model_selection", model_selection)
    prompt = prompt_entry.get()
    print (prompt)
    num_drawings = num_drawings_slider.get()
    guidance_scale = guidance_scale_slider.get()
    num_inference_steps = num_inference_steps_slider.get()

    for model_id in model_selection:
        model_name = model_id[-10:]
        print (model_id)
        print (model_name)

        if model_id != "DaleE":
            #pipe = StableDiffusionPipeline.from_pretrained(os.path.join (path_local,model_id ))
            pipe = DiffusionPipeline.from_pretrained(os.path.join (path_local,model_id )) #if it does not work 
            
            pipe = pipe.to("mps")
            pipe.enable_attention_slicing()

            for i in range(num_drawings):
                image = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
                short_name = prompt[:15]
                timestamp = datetime.now().strftime("%m%d-%H%M")
                file_name = sanitize_filename(f"{short_name}_{model_name}_{i+1}_{timestamp}.png")
                image.show(title=file_name)
                image.save(os.path.join(image_dir, file_name))
                print (file_name)
        else:
            for i in range(num_drawings):
                response = openai.Image.create(API_KEY, prompt=prompt, response_format="b64_json")
                img_bytes = b64decode(response["data"][0]["b64_json"])
                img = Image.open(BytesIO(img_bytes))
                short_name = prompt[:15]
                timestamp = datetime.now().strftime("%f")
                timestamp = timestamp[-4:]
                file_name = sanitize_filename(f"{short_name}_{model_name}_{i+1}_{timestamp}.png")
                img.show(title=file_name)
                with open(os.path.join(image_dir, file_name), "wb") as f:
                    f.write(img_bytes)

# Create a frame for center alignment
frame = tk.Frame(root)
frame.pack()

# Create the prompt entry
prompt_label = tk.Label(frame, text="Style and description of the Art you want to produce:")
prompt_label.pack()
prompt_entry = tk.Entry(frame, width= 60)
prompt_entry.pack()

# Create the model selection checkboxes
model_ids_vars = [tk.IntVar() for _ in range(len(model_ids))]
for i in range(len(model_ids)):
    model_checkbox = tk.Checkbutton(frame, text=model_ids[i], variable=model_ids_vars[i], anchor = 'w')
    model_checkbox.pack()

# Create the sliders with default values
num_drawings_slider = tk.Scale(frame, from_=1, to=1000, orient="horizontal", label="Number of Drawings",length=300)
num_drawings_slider.set(3)  # default value
num_drawings_slider.pack()

guidance_scale_slider = tk.Scale(frame, from_=1, to=10, orient="horizontal", label="Guidance Scale",length=300, resolution =0.1)
guidance_scale_slider.set(7.5)  # default value
guidance_scale_slider.pack()

num_inference_steps_slider = tk.Scale(frame, from_=1, to=100, orient="horizontal", label="Number of Inference Steps",length=300)
num_inference_steps_slider.set(25)  # default value
num_inference_steps_slider.pack()

# Create the generate button
generate_button = tk.Button(frame, text="Generate", command=generate_images)
generate_button.pack()

# Create a label for the image directory
image_dir_label = tk.Label(root, text="Drawings are saved in "+ image_dir)
image_dir_label.pack(side="bottom")

# Start the main loop
root.mainloop()