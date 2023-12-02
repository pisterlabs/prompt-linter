import urllib.request
from io import BytesIO
from PIL import Image, ImageTk
import openai
import tkinter as tk
import tkinter.messagebox

openai.api_key = "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"


def display_image(image_url):
    with urllib.request.urlopen(image_url) as url:
        image_data = url.read()

    image_stream = BytesIO(image_data)
    image = ImageTk.PhotoImage(Image.open(image_stream))
    image_label.configure(image=image)
    image_label.image = image


def get_image_url():
    try:
        response = openai.Image.create(
            prompt=input_field.get(),
            n=1,
            size="256x256"
        )
        image_url = response['data'][0]['url']
        return image_url
    except Exception as e:
        tk.messagebox.showerror(title="API Error", message=str(e))


def render_image():
    image_url = get_image_url()
    if image_url:
        display_image(image_url)


window = tk.Tk()
window.title("Image Generator")
window.geometry('450x400')

input_field = tk.Entry(window)
input_field.place(relx=0.5, rely=0.15, anchor='center')

generate_button = tk.Button(window, text="Generate", height=1, command=render_image)
generate_button.place(relx=0.5, rely=0.22, anchor='center')

image_frame = tk.Frame(window, bd=2, relief='groove')
image_frame.place(relx=0.5, rely=0.6, anchor='center', width=260, height=260)

image_label = tk.Label(image_frame)
image_label.pack(fill='both', expand=True)

window.mainloop()
