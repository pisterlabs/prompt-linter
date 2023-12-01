import gradio as gr
import keyboard
import requests
import json
import openai
from PIL import Image, ImageDraw, ImageFont, ImageOps
import textwrap
import random
import os
import time
from bing_image_downloader import downloader
import webbrowser
openai.api_key = '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'


def greet(cop, top, num):

    # Replace spaces in title with hyphens
    # Specify the path of the folder
    folder_path = "grads"

    # Get a list of all the file names in the folder
    file_names = os.listdir(folder_path)

    # Iterate over the file names and remove each file
    for file_name in file_names:
        os.remove(os.path.join(folder_path, file_name))
    # Download i mages of dogs
    downloader.download(f"{top}", limit=int(
        num), output_dir="images", adult_filter_off=True, force_replace=False)

    for i in range(1, int(num)+1):
        with open("news.txt", "r") as file:
            news = file.read().strip()
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"write a motivation line of or on {cop.upper()} except this: {news}  just write a motivation line from great person"}
            ]
        )
        OP = response['choices'][0]['message']['content']
        OP = OP.replace('"', '')
        with open("news.txt", "r") as file:
            if OP in file.read():
                print("This line has already been added to the file.")
            else:
                with open("news.txt", "a") as file:
                    file.write(OP + "\n")
                    print("Added the following line to the file: " + OP)

        # Make a request to the Unsplash API to get a random photo
                    image_dir = f"images/{top}/image_{int(num)}"
                    # Check if the image file exists with a .jpg or .png extension
                    if os.path.exists(f"{image_dir}.jpg"):
                        image_dir += ".jpg"
                    elif os.path.exists(f"{image_dir}.png"):
                        image_dir += ".png"
                    else:
                        print(f"No image found for {top} {i}")
                        continue  # Skip to the next iteration if no image is found
                    # Create an Image object with the local image file
                    img = Image.open(image_dir)

                    # Create a new white image of 600x600 size
                    new_img = Image.new('RGB', (600, 600), (255, 255, 255))

                    # Calculate the center position of the new image
                    x = (600 - img.width) // 2
                    y = (600 - img.height) // 2

                    # Paste the original image onto the new image at the center position
                    new_img.paste(img, (x, y))

    # Load the overlay image
                    overlay = Image.open('tranf.png')
                    overlay = overlay.resize((600, 600))
                    overlay = overlay.convert("RGBA")

                    # Paste the overlay image onto the main image
                    new_img.paste(overlay, (0, 0), overlay)
                    # Display the image

                    # Create an Image object with the randomly chosen background color

                    # Create a Draw object for drawing on the image
                    draw = ImageDraw.Draw(new_img)
                # Create a black layer with transparency of 30%

                    # reduced font size to 24
                    font = ImageFont.truetype('Muroslant.otf', size=23)

                    # Define the text to draw
                    text = f"{OP}"

                    # Wrap the text to fit within the image width
                    wrapped_text = textwrap.wrap(text, width=50)

                    # Calculate the position of the text in the center of the image
                    x = (new_img.width - draw.textbbox((0, 0),
                                                       wrapped_text[0], font=font)[2]) // 2
                    y = (new_img.height -
                         font.getsize(wrapped_text[0])[1] * len(wrapped_text)) // 2

                    # Draw each line of wrapped text on the image
                    for line in wrapped_text:
                        draw.text((x, y), line, fill='white',
                                  font=font, outline='black')
                        # draw.text((x, y), line, fill='black', font=font, stroke_width=1, stroke_fill='white', align='center')
                        y += draw.textsize(line, font=font)[1]

                    # Get the news item's title
                    title = f"grads"

                    # Create a folder to store the images if it doesn't exist
                    if not os.path.exists(title):
                        os.makedirs(title)

                    # Save the image with a unique filename
                    i = 1
                    while True:
                        img_filename = f"{title}/image{i}.png"
                        if not os.path.exists(img_filename):
                            break
                        i += 1

                    new_img.save(img_filename)
                    print(f"Saved image as {img_filename}")
                    return new_img


demo = gr.Interface(fn=greet, inputs=[
                    "text", "text", gr.inputs.Number(default=1)], outputs="image")


webbrowser.open("http://127.0.0.1:7860")
demo.launch()
