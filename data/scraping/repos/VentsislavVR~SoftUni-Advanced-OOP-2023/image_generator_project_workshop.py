import os.path
import urllib.request
from io import BytesIO
import openai
import tkinter as tk
from PIL import Image, ImageTk
import re
import random

openai.api_key = "API KEY"
MEDIA_FOLDER = "images/"


def convert_to_image_object(image_url):
    with urllib.request.urlopen(image_url) as url:
        image_data = url.read()

    image_stream = BytesIO(image_data)
    image = Image.open(image_stream)

    return image


def display_image(image):
    tk_image = ImageTk.PhotoImage(image)

    image_label.configure(image=tk_image)
    image_label.image = tk_image


def save_image(image, path):
    while os.path.isfile(path):
        image_name = re.match(r"images/(.*?)\.jpg",path)[1]
        new_name = image_name + str(random.randint(1,1_000_000))
        path = path.replace(image_name,new_name)

    image.save(path)


def get_image_url():
    response = openai.Image.create(
        prompt=input_field.get(),
        n=1,
        size="256x256"
    )
    image_url = response['data'][0]['url']

    print(image_url)

    return image_url


def render_image():
    global save_button

    try:
        save_button.destroy()
        image_url = get_image_url()
        image_name = "_".join(input_field.get().split()) + ".jpg"
        image = convert_to_image_object(image_url)
        display_image(image)
        save_button = tk.Button(
            window, text="Save",
            height=1,
            command=lambda: save_image(image, os.path.join(MEDIA_FOLDER, image_name)))
        save_button.place(x=350, y=17)
    except openai.error.InvalidRequestError:
        error_label = tk.Label(window, text="Prompt cannot be empty!", fg="red")
        error_label.place(x=175, y=50)


window = tk.Tk()
window.title("Image generator")
window.geometry("500x350")  # width * height

image_label = tk.Label(window)
image_label.place(x=125, y=70)

input_field = tk.Entry(window)
input_field.place(x=165, y=20)

generate_button = tk.Button(window,
                            text="Create",
                            height=1,
                            command=render_image
                            )
generate_button.place(x=300, y=17)
save_button = tk.Button(window, text="Save", height=1)

window.mainloop()
