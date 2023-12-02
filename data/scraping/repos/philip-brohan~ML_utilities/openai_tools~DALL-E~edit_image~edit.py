#!/usr/bin/env python

# Edit a digital image with the DALL-E api
# Doesn't work very well - tends to make an image that's either
#  almost the same as the original, or totally different.

import openai
import requests
import os
import sys
from PIL import Image

from openai_tools.config.access import get_key

openai.api_key = get_key()

prompt = "Watercolor in the style of John Constable"

# Load the image to edit
img_file = "./square_face.png"

# Make a mask file - same size as img file, but we only use the alpha channel
# Transparent pixels are regenerated, other pixels are kept
with open(img_file, "rb") as image_file:
    mask = Image.open(image_file)
    mask.load()

for h_i in range(mask.height):
    for w_i in range(mask.width):
        col = mask.getpixel((w_i, h_i))
        # print(col)
        # sys.exit(0)
        if col[0] > (col[2] * 1.1) and col[1] > (col[2] * 1.1):  # Is yellow
            mask.putpixel((w_i, h_i), (255, 0, 0, 0))
        else:
            mask.putpixel((w_i, h_i), (0, 255, 0, 255))

mask_file = "./mask.png"
mask.save(mask_file)

# call the OpenAI API
generation_response = openai.Image.create_edit(
    image=open(img_file, "rb"),
    mask=open(mask_file, "rb"),
    prompt=prompt,
    n=1,
    size="512x512",
    response_format="url",
)

generated_image_name = (
    "edited_image.png"  # any name you like; the filetype should be .png
)
generated_image_filepath = os.path.join(".", generated_image_name)
generated_image_url = generation_response["data"][0][
    "url"
]  # extract image URL from response
generated_image = requests.get(generated_image_url).content  # download the image

with open(generated_image_filepath, "wb") as image_file:
    image_file.write(generated_image)  # write the image to the file
