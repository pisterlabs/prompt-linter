import openai
import os
import requests
import glob
import re
from colorama import Fore, Style, init
import datetime
import base64
import json
import tkinter as tk
from PIL import Image, ImageTk

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def save_file(filepath, content):
    with open(filepath, 'a', encoding='utf-8') as outfile:
        outfile.write(content)

api_key = open_file('openaiapikey2.txt')
sd_api_key = open_file('sdapikey.txt')

def chatgpt(api_key, conversation, chatbot, user_input, temperature=0.8, frequency_penalty=0.2, presence_penalty=0):
    openai.api_key = api_key
    conversation.append({"role": "user","content": user_input})
    messages_input = conversation.copy()
    prompt = [{"role": "system", "content": chatbot}]
    messages_input.insert(0, prompt[0])

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        messages=messages_input)

    chat_response = completion['choices'][0]['message']['content']
    conversation.append({"role": "assistant", "content": chat_response})
    return chat_response

def generate_image(api_key, text_prompt, negative_prompts, height=512, width=512, cfg_scale=7, clip_guidance_preset="FAST_BLUE", steps=50, samples=1):
    api_host = 'https://api.stability.ai'
    engine_id = "stable-diffusion-xl-beta-v2-2-2"

    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/text-to-image",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        json={
            "text_prompts": [
                {
                    "text": text_prompt
                }
            ],
            "negative_prompts": [
                {
                    "text": negative_prompts
                }
            ],
            "cfg_scale": cfg_scale,
            "clip_guidance_preset": clip_guidance_preset,
            "height": height,
            "width": width,
            "samples": samples,
            "steps": steps,
        },
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()
    image_data = data["artifacts"][0]["base64"]

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    image_filename = os.path.join("SDimages", f"generated_image_{timestamp}.png")

    with open(image_filename, "wb") as f:
        f.write(base64.b64decode(image_data))

    return image_filename

def get_latest_image_filename(folder):
    list_of_files = glob.glob(f"{folder}/*.png")
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def update_ui_with_image(root, label):
    folder = "C:/Users/kris_/Python/sdspeed/SDimages"
    image_filename = get_latest_image_filename(folder)
    img = Image.open(image_filename)
    img = img.resize((1080, 1080), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    label.config(image=img)
    label.image = img
    root.update()

conversation = []
chatbot = "I am Photograph and Art Expert. I will assist you creating amazing prompts for all kinds of images."

root = tk.Tk()
root.title("Generated Images")
root.geometry("1200x1200")
label = tk.Label(root)
label.pack()

n = 25
for i in range(n):
    conversation = []
    chatbot = "I am Photograph and Art Expert. I will assist you creating amazing prompts for all kinds of images."
    text_prompt1 = open_file("prompts.txt")
    text_prompt = chatgpt(api_key, conversation, chatbot, text_prompt1)
    print(text_prompt)
    negative_prompts1 = open_file("nprompts.txt")
    negative_prompts = chatgpt(api_key, conversation, chatbot, negative_prompts1)
    print(negative_prompts)
    image_filename = generate_image(sd_api_key, text_prompt, negative_prompts)
    update_ui_with_image(root, label)

root.mainloop()