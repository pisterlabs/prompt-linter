#!/usr/bin/env python

# Vary a digital image with the DALL-E api

import openai
import requests
import os
from PIL import Image

from openai_tools.config.access import get_key

openai.api_key = get_key()


# Load the image to edit
img_file = "./square_face.png"


## call the OpenAI API
generation_response = openai.Image.create_variation(
    image=open(img_file, "rb"),
    n=1,
    size="512x512",
    response_format="url",
)

generated_image_name = (
    "varied_image.png"  # any name you like; the filetype should be .png
)
generated_image_filepath = os.path.join(".", generated_image_name)
generated_image_url = generation_response["data"][0][
    "url"
]  # extract image URL from response
generated_image = requests.get(generated_image_url).content  # download the image

with open(generated_image_filepath, "wb") as image_file:
    image_file.write(generated_image)  # write the image to the file
